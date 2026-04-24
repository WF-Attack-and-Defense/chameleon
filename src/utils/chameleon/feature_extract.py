# -1 is incoming packet, 1 is outgoing packet

import math
import sys
import numpy as np


def neighborhood(iterable):
    """
        # This function is used to get the neighborhood of a packet in a trace.

        Args:
            iterable: an iterable of packets.
        Returns:
            A generator of tuples, each containing the previous, current, and next packets in the trace.
    """
    iterator = iter(iterable)
    prev = (0)
    item = next(iterator)
    for nxt in iterator:
        yield (prev,item,nxt)
        prev = item
        item = nxt
    yield (prev,item,None)

def chunkIt(seq, num):
    """
        # This function is used to chunk a sequence into a list of chunks.

        Args:
            seq: the sequence to chunk.
            num: the number of chunks.
        Returns:
            A list of chunks, each containing num packets.
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def safe_stats(values):
    """
        # This function is used to calculate the safe statistics of a list of values.

        Args:
            values: the list of values to calculate the statistics of.
        Returns:
            A list of statistics.
    """
    if values is None or len(values) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    arr = np.array(values, dtype=float)
    return [
        float(arr.mean()),
        float(arr.std()),
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
        float(arr.min()),
        float(arr.max()),
    ]

def mean_std_max(values):
    """
        # This function is used to calculate the mean, standard deviation, and maximum of a list of values.

        Args:
            values: the list of values to calculate the statistics of.
        Returns:
            A list of statistics.
    """
    if not values:
        return [0.0, 0.0, 0.0]
    arr = np.array(values, dtype=float)
    return [float(arr.mean()), float(arr.std()), float(arr.max())]

def interarrival_stats(list_data, direction=None):
    """
        # This function is used to calculate the interarrival statistics of a list of packets.

        Args:
            list_data: the list of packets.
            direction: the direction of the packets.
        Returns:
            A list of statistics.
    """
    if direction is None:
        times = [t for t, _ in list_data]
    else:
        times = [t for t, d in list_data if d == direction]
    if len(times) < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    diffs = np.diff(times)
    return safe_stats(diffs)

def burst_features(list_data):
    """
        # This function is used to calculate the burst features of a list of packets.

        Args:
            list_data: the list of packets.
        Returns:
            A list of statistics.
    """
    if not list_data:
        return [0.0] * 15

    bursts_all = []
    bursts_in = []
    bursts_out = []

    prev_dir = list_data[0][1]
    current_len = 1
    for _, direction in list_data[1:]:
        if direction == prev_dir:
            current_len += 1
        else:
            bursts_all.append(current_len)
            if prev_dir == -1:
                bursts_in.append(current_len)
            else:
                bursts_out.append(current_len)
            prev_dir = direction
            current_len = 1

    bursts_all.append(current_len)
    if prev_dir == -1:
        bursts_in.append(current_len)
    else:
        bursts_out.append(current_len)

    num_bursts = float(len(bursts_all))
    change_rate = (num_bursts - 1.0) / float(len(list_data) - 1) if len(list_data) > 1 else 0.0
    total_packets = float(len(list_data))
    in_ratio = sum(bursts_in) / total_packets if total_packets else 0.0
    out_ratio = sum(bursts_out) / total_packets if total_packets else 0.0

    all_stats = safe_stats(bursts_all)
    in_stats = mean_std_max(bursts_in)
    out_stats = mean_std_max(bursts_out)

    return [num_bursts, change_rate] + all_stats[:5] + in_stats + out_stats + [in_ratio, out_ratio]

def direction_features(list_data, n_points=100):
    """
        # This function is used to calculate the direction features of a list of packets.

        Args:
            list_data: the list of packets.
            n_points: the number of points to calculate the features.
        Returns:
            A list of statistics.
    """
    if not list_data:
        return [0.0] * n_points
    dirs = np.array([d for _, d in list_data], dtype=float)
    cumul = np.cumsum(dirs)
    if len(cumul) == 1:
        return [float(cumul[0])] * n_points
    idx = np.linspace(0, len(cumul) - 1, n_points)
    samples = np.interp(idx, np.arange(len(cumul)), cumul)
    return [float(x) for x in samples]

def time_window_counts(list_data, n_windows=20):
    """
        # This function is used to calculate the time window counts of a list of packets.

        Args:
            list_data: the list of packets.
            n_windows: the number of windows.
        Returns:
            A list of statistics.
    """
    if not list_data:
        return [0.0] * (2 * n_windows)
    times = [t for t, _ in list_data]
    duration = times[-1]
    if duration <= 0:
        return [0.0] * (2 * n_windows)
    in_counts = [0] * n_windows
    out_counts = [0] * n_windows
    for t, d in list_data:
        w = int((t / duration) * n_windows)
        if w == n_windows:
            w -= 1
        if d == -1:
            in_counts[w] += 1
        else:
            out_counts[w] += 1
    return [float(x) for x in in_counts + out_counts]

def get_pkt_list(trace_data):
    """
        # This function is used to get the list of packets in a trace.

        Args:
            trace_data: the trace data.
        Returns:
            A list of packets.
    """
    first_line = trace_data[0]
    first_line = first_line.strip().split()

    first_time = float(first_line[0])
    dta = []
    for line in trace_data:
        b = line.strip().split()

        if float(b[1]) > 0:
            dta.append(((float(b[0])- first_time), 1))
        else:
            dta.append(((float(b[0]) - first_time), -1))
    return dta


def In_Out(list_data):
    """
        # This function is used to calculate the number of incoming and outgoing packets in a trace.

        Args:
            list_data: the list of packets.
        Returns:
            A tuple containing the number of incoming and outgoing packets.
    """
    In = []
    Out = []
    for p in list_data:
        if p[1] == -1:
            In.append(p)
        if p[1] == 1:
            Out.append(p)
    return In, Out

def inter_pkt_time(list_data):
    """
        Compute inter-packet times (diffs between consecutive packet timestamps).
    """
    times = [x[0] for x in list_data]
    if len(times) < 2:
        return []
    temp = []
    for elem, next_elem in zip(times, times[1:] + [times[0]]):
        temp.append(next_elem - elem)
    return temp[:-1]


def interarrival_times(list_data):
    """
        Interarrival times for In, Out, and Total packet streams.
    """
    In, Out = In_Out(list_data)
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    TOTAL = inter_pkt_time(list_data)
    return IN, OUT, TOTAL


def interarrival_maxminmeansd_stats(list_data):
    """
        Max, mean, std, and 75th percentile for In/Out/Total interarrival times.
    """
    interstats = []
    In, Out, Total = interarrival_times(list_data)
    if In and Out:
        avg_in = sum(In) / float(len(In))
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.append((
            max(In), max(Out), max(Total),
            avg_in, avg_out, avg_total,
            np.std(In), np.std(Out), np.std(Total),
            np.percentile(In, 75), np.percentile(Out, 75), np.percentile(Total, 75),
        ))
    elif Out and not In:
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.append((
            0, max(Out), max(Total),
            0, avg_out, avg_total,
            0, np.std(Out), np.std(Total),
            0, np.percentile(Out, 75), np.percentile(Total, 75),
        ))
    elif In and not Out:
        avg_in = sum(In) / float(len(In))
        avg_total = sum(Total) / float(len(Total))
        interstats.append((
            max(In), 0, max(Total),
            avg_in, 0, avg_total,
            np.std(In), 0, np.std(Total),
            np.percentile(In, 75), 0, np.percentile(Total, 75),
        ))
    else:
        interstats.append([0] * 12)
    return interstats


def time_percentile_stats(trace_data):
    """
        25th, 50th, 75th, 100th percentile of packet timestamps for In, Out, Total.
    """
    Total = get_pkt_list(trace_data)
    In, Out = In_Out(Total)
    In1 = [x[0] for x in In]
    Out1 = [x[0] for x in Out]
    Total1 = [x[0] for x in Total]
    STATS = []
    if In1:
        STATS.append(np.percentile(In1, 25))
        STATS.append(np.percentile(In1, 50))
        STATS.append(np.percentile(In1, 75))
        STATS.append(np.percentile(In1, 100))
    if not In1:
        STATS.extend([0] * 4)
    if Out1:
        STATS.append(np.percentile(Out1, 25))
        STATS.append(np.percentile(Out1, 50))
        STATS.append(np.percentile(Out1, 75))
        STATS.append(np.percentile(Out1, 100))
    if not Out1:
        STATS.extend([0] * 4)
    if Total1:
        STATS.append(np.percentile(Total1, 25))
        STATS.append(np.percentile(Total1, 50))
        STATS.append(np.percentile(Total1, 75))
        STATS.append(np.percentile(Total1, 100))
    if not Total1:
        STATS.extend([0] * 4)
    return STATS


def first_and_last_30_pkts_stats(trace_data):
    """
        Count of In/Out in first 30 and last 30 packets.
    """
    Total = get_pkt_list(trace_data)
    first30 = Total[:30]
    last30 = Total[-30:]
    first30in = [p for p in first30 if p[1] == -1]
    first30out = [p for p in first30 if p[1] == 1]
    last30in = [p for p in last30 if p[1] == -1]
    last30out = [p for p in last30 if p[1] == 1]
    return [len(first30in), len(first30out), len(last30in), len(last30out)]


def number_per_sec(trace_data):
    """
        Packets per second (by cumulative count per second): avg, std, median, min, max, and list.
    """
    Total = get_pkt_list(trace_data)
    if not Total:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []
    last_time = Total[-1][0]
    last_second = math.ceil(last_time)
    if last_second < 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []
    temp = []
    for i in range(1, int(last_second) + 1):
        c = sum(1 for p in Total if p[0] <= i)
        temp.append(c)
    l = []
    for prev, item, nxt in neighborhood(temp):
        x = item - prev
        l.append(x)
    if not l:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []
    avg_number_per_sec = sum(l) / float(len(l))
    return avg_number_per_sec, np.std(l), np.percentile(l, 50), min(l), max(l), l


def number_pkt_stats(trace_data):
    """
        # This function is used to calculate the number of incoming and outgoing packets in a trace.

        Args:
            trace_data: the trace data.
        Returns:
            A tuple containing the number of incoming and outgoing packets.
    """
    Total = get_pkt_list(trace_data)
    In, Out = In_Out(Total)
    return len(In), len(Out), len(Total)

#concentration of outgoing packets in chunks of 20 packets
def pkt_concentration_stats(trace_data):
    """
        # This function is used to calculate the concentration of outgoing packets in a trace.

        Args:
            trace_data: the trace data.
        Returns:
            A tuple containing the concentration of outgoing packets.
    """
    Total = get_pkt_list(trace_data)
    if not Total:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []
    chunks= [Total[x:x+20] for x in range(0, len(Total), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[1] == 1:
                c+=1
        concentrations.append(c)
    return np.std(concentrations), sum(concentrations)/float(len(concentrations)), np.percentile(concentrations, 50), min(concentrations), max(concentrations), concentrations

def avg_pkt_ordering_stats(trace_data):
    """
        # This function is used to calculate the average ordering of packets in a trace.

        Args:
            trace_data: the trace data.
        Returns:
            A tuple containing the average ordering of incoming and outgoing packets.
    """
    Total = get_pkt_list(trace_data)
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in Total:
        if p[1] == 1:
            temp1.append(c1)
        c1+=1
        if p[1] == -1:
            temp2.append(c2)
        c2+=1
    avg_in = sum(temp1)/float(len(temp1)) if temp1 else 0.0
    avg_out = sum(temp2)/float(len(temp2)) if temp2 else 0.0

    return avg_in, avg_out, np.std(temp1), np.std(temp2)

def perc_inc_out(trace_data):
    """
        # This function is used to calculate the percentage of incoming and outgoing packets in a trace.

        Args:
            trace_data: the trace data.
        Returns:
            A tuple containing the percentage of incoming and outgoing packets.
    """
    Total = get_pkt_list(trace_data)
    if not Total:
        return 0.0, 0.0
    In, Out = In_Out(Total)
    percentage_in = len(In)/float(len(Total))
    percentage_out = len(Out)/float(len(Total))
    return percentage_in, percentage_out


#If size information available add them in to function below
def TOTAL_FEATURES(trace_data, max_size=350):
    """
        # This function is used to calculate the total features of a trace.

        Args:
            trace_data: the trace data.
            max_size: the maximum size of the features.
        Returns:
            A tuple containing the total features.
    """
    list_data = get_pkt_list(trace_data)
    ALL_FEATURES = []

    # ------TIME--------
    intertimestats = [x for x in interarrival_maxminmeansd_stats(list_data)[0]]
    timestats = time_percentile_stats(trace_data)
    number_pkts = list(number_pkt_stats(trace_data))
    thirtypkts = first_and_last_30_pkts_stats(trace_data)
    stdconc, avgconc, medconc, minconc, maxconc, conc = pkt_concentration_stats(trace_data)
    avg_per_sec, std_per_sec, med_per_sec, min_per_sec, max_per_sec, per_sec = number_per_sec(trace_data)
    avg_order_in, avg_order_out, std_order_in, std_order_out = avg_pkt_ordering_stats(trace_data)
    perc_in, perc_out = perc_inc_out(trace_data)

    altconc = [sum(x) for x in chunkIt(conc, 70)]
    if len(altconc) == 70:
        altconc.append(0)
    alt_per_sec = [sum(x) for x in chunkIt(per_sec, 20)]
    if len(alt_per_sec) == 20:
        alt_per_sec.append(0)

    # TIME Features (interarrival and time percentile stats first)
    ALL_FEATURES.extend(intertimestats)
    ALL_FEATURES.extend(timestats)
    ALL_FEATURES.extend(number_pkts)
    ALL_FEATURES.extend(thirtypkts)
    ALL_FEATURES.append(stdconc)
    ALL_FEATURES.append(avgconc)
    ALL_FEATURES.append(avg_per_sec)
    ALL_FEATURES.append(std_per_sec)
    ALL_FEATURES.append(avg_order_in)
    ALL_FEATURES.append(avg_order_out)
    ALL_FEATURES.append(std_order_in)
    ALL_FEATURES.append(std_order_out)
    ALL_FEATURES.append(medconc)
    ALL_FEATURES.append(med_per_sec)
    ALL_FEATURES.append(min_per_sec)
    ALL_FEATURES.append(max_per_sec)
    ALL_FEATURES.append(maxconc)
    ALL_FEATURES.append(perc_in)
    ALL_FEATURES.append(perc_out)
    ALL_FEATURES.extend(altconc)
    ALL_FEATURES.extend(alt_per_sec)
    ALL_FEATURES.append(sum(altconc))
    ALL_FEATURES.append(sum(alt_per_sec))
    ALL_FEATURES.append(sum(intertimestats))
    ALL_FEATURES.append(sum(timestats))
    ALL_FEATURES.append(sum(number_pkts))

    times = [t for t, _ in list_data]
    duration = times[-1] if times else 0.0
    packet_rate = (len(list_data) / float(duration)) if duration > 0 else 0.0
    inter_all = interarrival_stats(list_data)
    inter_in = interarrival_stats(list_data, direction=-1)
    inter_out = interarrival_stats(list_data, direction=1)
    burst_features_data = burst_features(list_data)
    direction_features_data = direction_features(list_data, n_points=100)
    window_counts = time_window_counts(list_data, n_windows=20)

    ALL_FEATURES.append(duration)
    ALL_FEATURES.append(packet_rate)
    ALL_FEATURES.extend(inter_all)
    ALL_FEATURES.extend(inter_in)
    ALL_FEATURES.extend(inter_out)
    ALL_FEATURES.extend(burst_features_data)
    ALL_FEATURES.extend(direction_features_data)
    ALL_FEATURES.extend(window_counts)

    ALL_FEATURES.extend(conc)
    ALL_FEATURES.extend(per_sec)


    while len(ALL_FEATURES)<max_size:
        ALL_FEATURES.append(0)
    features = ALL_FEATURES[:max_size]

    return tuple(features)