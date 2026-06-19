""" k-nearest-neighbour composition of genomic tracks, per cell.

This module asks, for every cell of a Chromatin Tracing Experiment (CTE), how
much of a locus' immediate spatial neighbourhood is occupied by loci of a given
'target' track sitting on *other* chromosome-copies. It is the threshold-free,
scale-free replacement for the old rank-proximity matrix engine.

MOTIVATION
----------
Physical 3D distances and fixed-radius contacts are confounded by nuclear
volume: a 1 um sphere encloses far more chromatin in a small/dense nucleus than
in a swollen one, so comparing distances or contacts across conditions (e.g.
cell-cycle states) needs a volume normalisation that is itself fraught. A
k-nearest-neighbour count sidesteps this entirely: 'the k closest loci' is the
same set of loci no matter how the nucleus is globally rescaled, because a
monotonic rescaling of every distance leaves their ordering - and hence the
identity of the k nearest - unchanged. The neighbourhood is therefore a
scale-free, threshold-free probe of local chromatin composition.

DEFINITION
----------
A 'track' is a named set of genomic intervals (e.g. the first 5 Mb of every
chromosome). A cell's loci are partitioned into chromosome-copies, each
identified by (chromosome, traceID). For a pair of tracks (src, tgt) we look at
every source locus q in the src track and at its k nearest neighbours, taken
over the WHOLE pool of imaged loci (both same-chromosome and other-chromosome
loci fill the k slots - so a locus buried inside its own chromosome territory
spends its neighbourhood on its own copy and is correctly penalised).

A neighbour is a 'hit' if it is on the tgt track AND on a different
chromosome-copy than q:

    hit(q, s) = 1   iff   s in tgt-track  and  copy(s) != copy(q)            (1)

Only q's own copy is excluded; the homolog (the other copy of q's chromosome)
counts as a hit, because two copies of chr2 near chr1-copyA is genuinely twice
as telling as one. We summarise the hits among q's k nearest two ways:

    composition(q, k) = #{ distinct copies contributing >=1 hit }           (2)
    raw(q, k)         = #{ hit neighbours }                                  (3)

(2) is the headline metric: how many *different* partner chromosome-copies have
planted themselves in q's local neighbourhood (a partner copy that fills five
of the k slots counts once). (3) is the un-deduplicated hit count, kept along
for free. Both are stored per source locus, for every k in a sweep; aggregation
over source loci and the cross-state / control-track comparison are left to
downstream analysis.

OUTPUTS
-------
A single HDF5 file with the Index/Genome, the cell labels, and the run metadata
(tracks, pairs, neighbours) as root attributes, plus one group per cell:

    cells/<cellID>/chrom            (M,)            chromosome of each chrom-copy
    cells/<cellID>/trace            (M,)            traceID of each chrom-copy
    cells/<cellID>/copy_size        (M,)            #spots in each chrom-copy
    cells/<cellID>/<src>__<tgt>/
        n_src                       scalar          #source-track loci
        n_tgt                       scalar          #target-track loci
        src_copy                    (n_src,)        copy index of each source locus
        composition                 (n_src, nk)     metric (2) per locus, per k
        raw                         (n_src, nk)     metric (3) per locus, per k
"""

import os
import json
import warnings
import h5py
import numpy as np
from scipy.spatial import cKDTree
from ..cte import ChromatinTracingExperiment
from .. import parallel


# Default neighbourhood sizes (the k sweep) if the config does not specify them.
DEFAULT_NEIGHBORS = [25, 50, 100, 250, 500]


# SPOT GATHERING AND TRACK MASKS


def gather_cell_spots(cte: ChromatinTracingExperiment, cellID: str) -> dict:
    """ Collect all imaged loci of a cell into flat arrays.

    Each chromosome-copy is identified by (chromosome, traceID). All spots of
    all chrom-copies are concatenated into single arrays, with a per-spot index
    into the list of chrom-copies.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)

    Returns:
        dict with keys:
            'X'           (N, 3) float64    spot coordinates (um)
            'start'       (N,)   int64      genomic start (bp) of each spot
            'spot_copy'   (N,)   int        index of each spot's chrom-copy
            'spot_chromid'(N,)   int        index of each spot's chromosome
            'copy_chrom'  (M,)   '<U16'     chromosome of each chrom-copy
            'copy_trace'  (M,)   '<U16'     traceID of each chrom-copy
            'copy_chromid'(M,)   int        chromosome index of each chrom-copy
            'copy_size'   (M,)   int        #spots in each chrom-copy
        Returns None if the cell has no spots.
    """

    trace_map = cte.get_trace_hashmap(cellID)  # {chrom: {traceID_str: idx}}

    Xs, starts, spot_copy = [], [], []
    copy_chrom, copy_trace = [], []

    copy_idx = 0
    for chrom in trace_map:
        for traceID in trace_map[chrom]:
            d = cte.get_data(cellID, chrom, traceID, format='numpy')
            x = np.asarray(d['xs'], dtype=np.float64)
            if x.size == 0:
                continue
            y = np.asarray(d['ys'], dtype=np.float64)
            z = np.asarray(d['zs'], dtype=np.float64)
            # Drop any spot with a NaN coordinate (defensive; projected CTEs are clean)
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not good.any():
                continue
            Xs.append(np.column_stack((x[good], y[good], z[good])))
            starts.append(np.asarray(d['starts'], dtype=np.int64)[good])
            spot_copy.append(np.full(int(good.sum()), copy_idx, dtype=np.int64))
            copy_chrom.append(str(chrom))
            copy_trace.append(str(traceID))
            copy_idx += 1

    if copy_idx == 0:
        return None

    copy_chrom = np.array(copy_chrom)
    copy_trace = np.array(copy_trace)
    # Integer chromosome id per chrom-copy (shared id for the two homologs)
    uniq_chroms, copy_chromid = np.unique(copy_chrom, return_inverse=True)

    spot_copy = np.concatenate(spot_copy)
    return {
        'X':            np.concatenate(Xs),
        'start':        np.concatenate(starts),
        'spot_copy':    spot_copy,
        'spot_chromid': copy_chromid[spot_copy],
        'copy_chrom':   copy_chrom,
        'copy_trace':   copy_trace,
        'copy_chromid': copy_chromid,
        'copy_size':    np.bincount(spot_copy, minlength=copy_idx),
    }


def build_track_masks(spots: dict, tracks: dict) -> dict:
    """ For each track, a boolean mask over spots: True if the spot's
    (chromosome, start) falls inside one of the track's intervals.

    Args:
        spots (dict): output of gather_cell_spots
        tracks (dict): {track_name: [(chrom, start_bp, end_bp), ...]}.
            A spot belongs to the track if its chromosome matches and its
            genomic start lies in [start_bp, end_bp).

    Returns:
        dict {track_name: (N,) bool}
    """

    copy_chrom = spots['copy_chrom']
    spot_chrom = copy_chrom[spots['spot_copy']]   # (N,) chromosome string per spot
    start = spots['start']
    N = start.size

    masks = {}
    for name, intervals in tracks.items():
        m = np.zeros(N, dtype=bool)
        for chrom, s0, s1 in intervals:
            m |= (spot_chrom == str(chrom)) & (start >= int(s0)) & (start < int(s1))
        masks[name] = m
    return masks


# PER-CELL k-NEAREST-NEIGHBOUR COMPOSITION


def compute_cell_composition(spots: dict, masks: dict, pairs: list,
                             neighbors: list) -> dict:
    """ k-nearest-neighbour composition for one cell, for every track pair.

    For each source-track locus we find its k nearest neighbours over the whole
    pool of loci, count how many distinct *other* chrom-copies of the target
    track appear among them (composition, deduplicated by copy) and how many
    target neighbours there are in total (raw, not deduplicated), for every k.

    Args:
        spots (dict): output of gather_cell_spots.
        masks (dict): output of build_track_masks (must cover every track in pairs).
        pairs (list): [(src_track, tgt_track), ...].
        neighbors (list): the k sweep, e.g. [25, 50, 100, 250, 500].

    Returns:
        dict {f'{src}__{tgt}': {
            'n_src' (int), 'n_tgt' (int),
            'src_copy'    (n_src,)     copy index of each source locus,
            'composition' (n_src, nk)  distinct partner copies among k nearest,
            'raw'         (n_src, nk)  target neighbours among k nearest,
        }} where nk = len(neighbors).
    """

    X = spots['X']                       # (N, 3)
    spot_copy = spots['spot_copy']       # (N,)
    N = X.shape[0]
    M = spots['copy_size'].size          # number of chrom-copies

    ks = np.sort(np.asarray(neighbors, dtype=np.int64))   # (nk,) ascending k values
    kmax = int(ks[-1])

    # The neighbours come from the WHOLE pool (intra + inter), so the tree holds
    # every spot; but only source-track loci ever need their neighbourhood
    # queried. Query the union of all source tracks once and index back per pair
    # (sources are a small subset of the pool, so this is a big saving at fine
    # resolution where N is large).
    tree = cKDTree(X)
    src_any = np.zeros(N, dtype=bool)
    for src, _ in pairs:
        src_any |= masks[src]
    qry_idx = np.where(src_any)[0]                   # (n_qry,) global spot indices
    g2q = np.full(N, -1, dtype=np.int64)             # global spot -> row in query block
    g2q[qry_idx] = np.arange(qry_idx.size)

    nbr = np.empty((qry_idx.size, 0), dtype=np.int64)
    nbr_copy = nbr
    if qry_idx.size > 0:
        # Query one extra neighbour: each query point sits in the tree, so its
        # own self is the nearest (distance 0); drop that first column.
        kq = min(kmax + 1, N)
        _, nbr = tree.query(X[qry_idx], k=kq)        # (n_qry, kq) nearest first
        nbr = np.atleast_2d(nbr)[:, 1:]              # (n_qry, kq-1) drop self column
        nbr_copy = spot_copy[nbr]                    # (n_qry, kq-1) neighbour copies

    result = {}
    for src, tgt in pairs:
        src_idx = np.where(masks[src])[0]            # (n_src,) source-track loci
        n_src = src_idx.size
        n_tgt = int(masks[tgt].sum())

        # Allocate even when a track is empty, so the storage layout is uniform.
        composition = np.zeros((n_src, ks.size), dtype=np.int64)
        raw = np.zeros((n_src, ks.size), dtype=np.int64)

        if n_src > 0:
            q = g2q[src_idx]                         # (n_src,) rows into the query block
            own_copy = spot_copy[src_idx][:, None]   # (n_src, 1) source's own copy
            sub_nbr = nbr[q]                         # (n_src, kq-1) neighbour indices
            sub_copy = nbr_copy[q]                   # (n_src, kq-1) neighbour copies
            is_tgt = masks[tgt][sub_nbr]             # (n_src, kq-1) neighbour on tgt track?
            # A hit is a target neighbour on a DIFFERENT copy than the source.
            # The homolog (other copy of the same chromosome) counts; only the
            # source's own copy is excluded.
            hit = is_tgt & (sub_copy != own_copy)    # (n_src, kq-1)
            rows = np.arange(n_src)

            for j, k in enumerate(ks):
                hit_k = hit[:, :k]                   # (n_src, k) hits among k nearest
                raw[:, j] = hit_k.sum(axis=1)
                # Distinct partner copies: scatter the hit copies into a
                # (n_src, M) table, then count the occupied columns per source.
                seen = np.zeros((n_src, M), dtype=bool)
                rr = np.broadcast_to(rows[:, None], hit_k.shape)[hit_k]
                cc = sub_copy[:, :k][hit_k]
                seen[rr, cc] = True
                composition[:, j] = seen.sum(axis=1)

        result[f'{src}__{tgt}'] = {
            'n_src':       n_src,
            'n_tgt':       n_tgt,
            'src_copy':    spot_copy[src_idx].astype(np.int64),
            'composition': composition,
            'raw':         raw,
        }
    return result


# PARALLEL NODE / REDUCE FUNCTIONS


def func_node(cellID: str, cte_name: str, _, config: dict) -> dict:
    """ Node task: compute the kNN composition for a single cell.

    Args:
        cellID (str)
        cte_name (str)
        _ : not used (scf), just to match the signature of the parallel call.
        config (dict): see run_knn_composition.

    Returns:
        dict: node result, or None if the cell has no usable spots.
    """
    cte = ChromatinTracingExperiment(cte_name, 'r')
    spots = gather_cell_spots(cte, cellID)
    cte.close()
    if spots is None:
        return None

    neighbors = config.get('neighbors', DEFAULT_NEIGHBORS)
    masks = build_track_masks(spots, config['tracks'])
    comp = compute_cell_composition(spots, masks,
                                    [tuple(p) for p in config['pairs']], neighbors)

    return {
        'copy_chrom': spots['copy_chrom'],
        'copy_trace': spots['copy_trace'],
        'copy_size':  spots['copy_size'],
        'pairs':      comp,
    }


def reduce_initialization(parallelIDs, cte_name: str, _, config: dict) -> None:
    """ Create the output HDF5 file: store the Index/Genome, the cell labels,
    and the run metadata (tracks, pairs, neighbours) as root attributes.
    """
    cte = ChromatinTracingExperiment(cte_name, 'r')
    index = cte.index
    cell_labels = np.asarray(cte.cell_labels).astype('S')
    cte.close()

    with h5py.File(config['filename'], 'w') as h5:
        index.save(h5)
        h5.create_dataset('cell_labels', data=cell_labels)
        # track intervals: cast to plain lists of [chrom, int, int] for JSON
        tracks_json = {name: [[str(c), int(s0), int(s1)] for c, s0, s1 in iv]
                       for name, iv in config['tracks'].items()}
        h5.attrs['tracks'] = json.dumps(tracks_json)
        h5.attrs['pairs'] = json.dumps([list(p) for p in config['pairs']])
        h5.attrs['neighbors'] = json.dumps(list(config.get('neighbors', DEFAULT_NEIGHBORS)))
        h5.create_group('cells')


def reduce_update(cellID: str, _result, node_result: dict, cte_name: str,
                  _scf, config: dict) -> None:
    """ Append one cell's composition arrays to the output HDF5 file. """
    if node_result is None:
        return None

    with h5py.File(config['filename'], 'a') as h5:
        cg = h5['cells'].create_group(str(cellID))
        cg.create_dataset('chrom', data=node_result['copy_chrom'].astype('S'))
        cg.create_dataset('trace', data=node_result['copy_trace'].astype('S'))
        cg.create_dataset('copy_size', data=node_result['copy_size'].astype(np.int64))

        for key, entry in node_result['pairs'].items():
            pg = cg.create_group(key)
            pg.create_dataset('n_src', data=np.int64(entry['n_src']))
            pg.create_dataset('n_tgt', data=np.int64(entry['n_tgt']))
            pg.create_dataset('src_copy', data=entry['src_copy'].astype(np.int64))
            pg.create_dataset('composition', data=entry['composition'].astype(np.int64),
                              compression='gzip')
            pg.create_dataset('raw', data=entry['raw'].astype(np.int64),
                              compression='gzip')


# MAIN ENTRY POINT


required_keys = {
    'tracks':    {'type': dict},
    'pairs':     {'type': list},
    'filename':  {'type': str},
    'neighbors': {'type': list, 'optional': True},
}


def run_knn_composition(cte: ChromatinTracingExperiment, config: dict) -> None:
    """ Compute the k-nearest-neighbour track composition, per cell,
    parallelised across cells.

    Args:
        cte (ChromatinTracingExperiment): the experiment providing the loci
            coordinates. Use a projected CTE (one consensus point per genomic
            bin) so the neighbourhood samples the genome uniformly; an imputed
            CTE additionally gives the same loci in every cell, which makes the
            composition counts directly comparable across cells.
        config (dict):
            - 'tracks' (dict): {track_name: [(chrom, start_bp, end_bp), ...]}.
                A spot belongs to a track if its (chrom, start) is in an interval.
            - 'pairs' (list): [(src_track, tgt_track), ...] to compute.
            - 'filename' (str): output HDF5 path.
            - 'neighbors' (list, optional): the k sweep. Default DEFAULT_NEIGHBORS.
            - 'parallel' (dict): controller config (see imgtools.parallel).
    """
    # Validate the neighbour sweep early, with a clear message.
    neighbors = config.get('neighbors', DEFAULT_NEIGHBORS)
    if any(int(k) <= 0 for k in neighbors):
        raise ValueError(f'neighbors must be positive integers. Got {neighbors}.')

    parallel.control_func(
        cte, None,
        config, required_keys,
        func_node, reduce_initialization, reduce_update,
        mode='cell'
    )
