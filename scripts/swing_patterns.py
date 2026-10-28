"""Causal daily swing-pattern research rules; numeric thresholds are v1 policy."""
import numpy as np
import pandas as pd

from scripts.chart_patterns import PatternSignal

ASCENDING_TRIANGLE = 'ascending_triangle_breakout_v1'
DOUBLE_BOTTOM = 'double_bottom_breakout_v1'
CUP_HANDLE = 'cup_handle_breakout_v1'
INVERSE_HEAD_SHOULDERS = 'inverse_head_shoulders_breakout_v1'
STRATEGIES = (ASCENDING_TRIANGLE, DOUBLE_BOTTOM, CUP_HANDLE, INVERSE_HEAD_SHOULDERS)


def _input(bars):
    columns = ['open', 'high', 'low', 'close', 'volume']
    if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.hasnans or
        not bars.index.is_monotonic_increasing or bars.index.normalize().has_duplicates or
        not set(columns) <= set(bars.columns) or len(bars) < 31):
        return None
    frame = bars.iloc[-141:][columns].apply(pd.to_numeric, errors='coerce')
    values = frame.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values <= 0).any() or
        (frame.high < frame[['open', 'close']].max(axis=1)).any() or
        (frame.low > frame[['open', 'close']].min(axis=1)).any()):
        return None
    current = frame.iloc[-1]
    ratio = float(current.volume/frame.volume.iloc[-21:-1].mean())
    if current.close <= current.open or ratio < 1.5:
        return None
    return frame.iloc[:-1], current, ratio


def _pivots(values, low=False):
    """Two completed bars on each side; breakout bar never confirms a pivot."""
    values=np.asarray(values)
    center=values[2:-2]
    if not len(center):return []
    neighbors=(values[:-4],values[1:-3],values[3:-1],values[4:])
    mask=np.ones(len(center),dtype=bool)
    for neighbor in neighbors:
        mask &= center<neighbor if low else center>neighbor
    return (np.flatnonzero(mask)+2).tolist()


def _signal(strategy_id, bars, current, resistance, floor, height, metrics):
    entry=float(current.close)
    target=float(resistance+height)
    if not (resistance*1.001 < entry <= resistance*1.05 and 0 < floor*.999 < entry < target):
        return None
    return PatternSignal(pattern=strategy_id,signal_time=bars.index[-1].isoformat(),
        entry=entry,resistance=float(resistance),invalidation=float(floor*.999),target=target,metrics=metrics)


def detect_ascending_triangle(bars):
    supplied=_input(bars)
    if supplied is None:return None
    prior,current,volume=supplied
    highs=prior.high.to_numpy();lows=prior.low.to_numpy()
    closes=prior.close.to_numpy();volumes=prior.volume.to_numpy()
    if current.close<=float(highs[-15:].max())*1.001:return None
    # Longest qualifying formation wins, deterministically.
    for length in range(min(60,len(prior)),14,-1):
        offset=len(prior)-length
        formation_highs=highs[offset:];formation_lows=lows[offset:]
        peaks=_pivots(formation_highs);troughs=_pivots(formation_lows,low=True)
        resistance=float(formation_highs.max())
        touches=[i for i in peaks if formation_highs[i]>=resistance*.99]
        if len(touches)<2 or touches[-1]-touches[0]<5 or len(troughs)<2:continue
        if not any(touches[0]<i<touches[-1] for i in troughs):continue
        reaction_lows=formation_lows[troughs]
        if (np.any(np.diff(reaction_lows)<=resistance*.005) or
            reaction_lows[-1]-reaction_lows[0]<resistance*.02 or
            troughs[-1]-troughs[0]<5):continue
        # No breakdown through the last confirmed support after it formed.
        floor=float(reaction_lows[-1])
        if float(formation_lows[troughs[-1]+1:].min())<floor:continue
        half=length//2
        formation_volumes=volumes[offset:]
        if formation_volumes[half:].mean()>formation_volumes[:half].mean():continue
        if (closes[offset:]>resistance*1.001).any():continue
        result=_signal(ASCENDING_TRIANGLE,bars,current,resistance,floor,resistance-float(formation_lows.min()),
            dict(formation_bars=length,resistance_touches=len(touches),rising_troughs=len(troughs),volume_ratio=volume))
        if result:return result
    return None


def detect_double_bottom(bars):
    supplied=_input(bars)
    if supplied is None:return None
    prior,current,volume=supplied
    lows=prior.low.to_numpy();troughs=_pivots(lows,low=True)
    for second in reversed(troughs):
        if len(prior)-1-second>20:continue
        for first in reversed([i for i in troughs if 20<=second-i<=80 and i>=20]):
            bottom=float(min(lows[first],lows[second]))
            if abs(lows[first]-lows[second])/bottom>.03:continue
            preceding=prior.iloc[first-20:first]
            if float(preceding.close.iloc[0])<lows[first]*1.10:continue
            if float(preceding.low.min())<lows[first]:continue
            between=prior.iloc[first+1:second]
            resistance=float(between.high.max())
            if resistance/bottom-1<.10:continue
            if float(prior.low.iloc[first:second+1].min())<bottom*.97:continue
            if float(prior.low.iloc[second+1:].min())<bottom*.97:continue
            if (prior.close.iloc[second+1:]>resistance*1.001).any():continue
            if prior.volume.iloc[second-2:second+3].mean()>prior.volume.iloc[first-2:first+3].mean():continue
            result=_signal(DOUBLE_BOTTOM,bars,current,resistance,bottom,resistance-bottom,
                dict(trough_separation_bars=second-first,first_trough=float(lows[first]),
                     second_trough=float(lows[second]),volume_ratio=volume))
            if result:return result
    return None


def detect_cup_handle(bars):
    supplied=_input(bars)
    if supplied is None:return None
    prior,current,volume=supplied
    highs=prior.high.to_numpy();lows=prior.low.to_numpy()
    closes=prior.close.to_numpy();volumes=prior.volume.to_numpy()
    if current.close<=float(highs[-5:].max())*1.001:return None
    for handle_length in range(15,4,-1):
        end=len(prior)-handle_length
        handle_highs=highs[end:];handle_lows=lows[end:]
        handle_closes=closes[end:];handle_volumes=volumes[end:]
        for cup_length in range(min(100,end-20),29,-1):
            start=end-cup_length
            cup_highs=highs[start:end];cup_lows=lows[start:end]
            cup_closes=closes[start:end];cup_volumes=volumes[start:end]
            left,right=float(cup_highs[0]),float(cup_highs[-1])
            if abs(left-right)/min(left,right)>.03:continue
            resistance=max(left,right,float(handle_highs.max()))
            bottom=float(cup_lows.min());height=resistance-bottom
            if not .12<=height/resistance<=.40:continue
            bottom_index=int(np.argmin(cup_lows))
            if not .3*cup_length<=bottom_index<=.7*cup_length:continue
            # Require time near the lows: a one-bar V does not qualify as a cup.
            if (cup_closes<bottom+height*.5).sum()<cup_length*.25:continue
            radius=max(2,int(cup_length*.1))
            center=cup_closes[max(0,bottom_index-radius):bottom_index+radius+1]
            if float(center.max()-center.min())>height*.15:continue
            if (float(cup_closes[0])<float(closes[start-20])*1.10 or
                float(cup_closes[-1])<resistance*.97):continue
            floor=float(handle_lows.min())
            if floor<resistance-height/3 or floor>=resistance:continue
            if handle_volumes.mean()>cup_volumes.mean():continue
            if (handle_closes>resistance*1.001).any():continue
            if float(handle_closes[-1])>=float(handle_closes[0]):continue
            result=_signal(CUP_HANDLE,bars,current,resistance,floor,height,
                dict(cup_bars=cup_length,handle_bars=handle_length,cup_depth_ratio=height/resistance,volume_ratio=volume))
            if result:return result
    return None


def detect_inverse_head_shoulders(bars):
    supplied=_input(bars)
    if supplied is None:return None
    prior,current,volume=supplied
    lows=prior.low.to_numpy();troughs=_pivots(lows,low=True)
    triples=list(zip(troughs,troughs[1:],troughs[2:]))
    for left,head,right in reversed(triples):
        if left<20 or not 5<=head-left<=40 or not 5<=right-head<=40:continue
        if not 20<=right-left<=80 or len(prior)-1-right>20:continue
        shoulder=float(min(lows[left],lows[right]))
        if abs(lows[left]-lows[right])/shoulder>.03 or lows[head]>shoulder*.95:continue
        if float(prior.close.iloc[left-20])<lows[left]*1.10:continue
        if float(prior.low.iloc[left-20:left].min())<lows[left]:continue
        first_neck=float(prior.high.iloc[left+1:head].max())
        second_neck=float(prior.high.iloc[head+1:right].max())
        if abs(first_neck-second_neck)/min(first_neck,second_neck)>.03:continue
        resistance=max(first_neck,second_neck)
        if resistance/shoulder-1<.05:continue
        if float(prior.low.iloc[right+1:].min())<lows[right]*.97:continue
        if (prior.close.iloc[right+1:]>resistance*1.001).any():continue
        if prior.volume.iloc[right-2:right+3].mean()>prior.volume.iloc[left-2:left+3].mean():continue
        result=_signal(INVERSE_HEAD_SHOULDERS,bars,current,resistance,float(lows[right]),resistance-float(lows[head]),
            dict(shoulder_span_bars=right-left,left_shoulder=float(lows[left]),head=float(lows[head]),
                 right_shoulder=float(lows[right]),volume_ratio=volume))
        if result:return result
    return None


DETECTORS={ASCENDING_TRIANGLE:detect_ascending_triangle,DOUBLE_BOTTOM:detect_double_bottom,
           CUP_HANDLE:detect_cup_handle,INVERSE_HEAD_SHOULDERS:detect_inverse_head_shoulders}
