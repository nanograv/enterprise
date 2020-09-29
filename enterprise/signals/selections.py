# selections.py
"""Contains various selection functions to mask parameters by backend flags,
time-intervals, etc."""

from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import inspect

import numpy as np


def call_me_maybe(obj):
    """See `here`_ for description.

    .. _here: https://www.youtube.com/watch?v=fWNaR-rxAic
    """
    return obj() if hasattr(obj, "__call__") else obj


def selection_func(func):
    try:
        funcargs = inspect.getfullargspec(func).args
    except:
        funcargs = inspect.getargspec(func).args

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        targs = list(args)

        # check for mask
        mask = kwargs.get("mask", Ellipsis)
        if "mask" in kwargs:
            del kwargs["mask"]

        if len(targs) < len(funcargs) and "psr" in kwargs:
            psr = kwargs["psr"]

            for funcarg in funcargs[len(args) :]:
                if funcarg not in kwargs and hasattr(psr, funcarg):
                    attr = call_me_maybe(getattr(psr, funcarg))
                    if isinstance(attr, np.ndarray) and getattr(mask, "shape", [0])[0] == len(attr):
                        targs.append(attr[mask])
                    else:
                        targs.append(attr)

        if "psr" in kwargs and "psr" not in funcargs:
            del kwargs["psr"]

        return func(*targs, **kwargs)

    return wrapper


def Selection(func):
    """Class factory for TOA selection."""

    class Selection(object):
        def __init__(self, psr):
            self._psr = psr

        @property
        def masks(self):
            return selection_func(func)(psr=self._psr)

        def _get_masked_array_dict(self, masks, arr):
            return {key: val * arr for key, val in masks.items()}

        def __call__(self, parname, parameter, arr=None):
            params, kmasks = {}, {}
            for key, val in self.masks.items():
                kname = "_".join([key, parname]) if key else parname
                pname = "_".join([self._psr.name, kname])
                params.update({kname: parameter(pname)})
                kmasks.update({kname: val})

            if arr is not None:
                ma = self._get_masked_array_dict(kmasks, arr)
                ret = (params, ma)
            else:
                ret = params, kmasks
            return ret

    return Selection


# SELECTION FUNCTIONS


def cut_half(toas):
    """Selection function to split by data segment"""
    midpoint = (toas.max() + toas.min()) / 2
    return dict(zip(["t1", "t2"], [toas <= midpoint, toas > midpoint]))


def by_band(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    flagvals = np.unique(flags["B"])
    return {flagval: flags["B"] == flagval for flagval in flagvals}


def by_frontend(flags):
    """Selection function to split by frontend under -fe flag"""
    flagvals = np.unique(flags["fe"])
    return {flagval: flags["fe"] == flagval for flagval in flagvals}


def by_band(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    flagvals = np.unique(flags['B'])
    return {flagval: flags['B'] == flagval for flagval in flagvals}

def by_b_10cm(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    return {'10CM': np.char.lower(flags['B']) == '10cm'}

def by_b_20cm(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    return {'20CM': np.char.lower(flags['B']) == '20cm'}

def by_b_40cm(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    return {'40CM': np.char.lower(flags['B']) == '40cm'}

def by_b_50cm(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    return {'50CM': np.char.lower(flags['B']) == '50cm'}

def by_b_4050cm(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    sel_40b = np.char.lower(flags['B']) == '40cm'
    sel_50b = np.char.lower(flags['B']) == '50cm'
    return {'4050CM': sel_40b + sel_50b}

def by_b_1020cm(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    sel_10b = np.char.lower(flags['B']) == '10cm'
    sel_20b = np.char.lower(flags['B']) == '20cm'
    return {'1020CM': sel_10b + sel_20b}

def by_legacy(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    flagvals = np.unique(flags['legacy'])
    return {flagval: flags['legacy'] == flagval for flagval in flagvals}


def by_backend(backend_flags):
    """Selection function to split by backend flags."""
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def by_frequency(freqs):
    """Selection function to split by frequency bands"""
    return dict(zip(['low_freq', 'mid_freq', 'high_freq'],
                    [freqs <= 1000, (freqs > 1000)*(freqs <= 2000),
                     freqs > 2000]))


def nanograv_backends(backend_flags):
    """Selection function to split by NANOGRav backend flags only."""
    flagvals = np.unique(backend_flags)
    ngb = ['ASP', 'GASP', 'GUPPI', 'PUPPI']
    flagvals = filter(lambda x: any(map(lambda y: y in x, ngb)), flagvals)
    return {flagval: backend_flags == flagval for flagval in flagvals}

def backends_except_ppta_legacy(backend_flags):
    flagvals = np.unique(backend_flags)
    ngb = ['20cm_afb','20cm_cpsr2','20cm_fptm']
    accepted_flags = np.array([ff for ff in flagvals if ff not in ngb])
    return {flagval: backend_flags == flagval for flagval in accepted_flags}

def band_50cm(freqs):
    """Selection only obs <700MHz"""
    return dict(zip(['50cm'], [freqs < 700]))


def band_40cm(freqs):
    """Selection only obs 700MHz <= f < 1000MHz"""
    return dict(zip(['40cm'], [(freqs >= 700)*(freqs < 1000)]))


def band_4050cm(freqs):
    """Selection only obs f < 1000MHz"""
    return dict(zip(['4050cm'], [freqs < 1000]))

def band_1020cm(freqs):
    """Selection only obs f >= 1000MHz"""
    return dict(zip(['1020cm'], [freqs >= 1000]))

def band_20cm(freqs):
    """Selection only obs 1000 <= f < 2000MHz"""
    return dict(zip(['20cm'], [(freqs >= 1000)*(freqs < 2000)]))


def band_10cm(freqs):
    """Selection only obs >=2000MHz"""
    return dict(zip(['10cm'], [freqs > 2000]))

def system_caspsr(backend_flags):
    seldict = dict()
    seldict['caspsr'] = np.array([True if 'caspsr' in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def system_cpsr2(backend_flags):
    seldict = dict()
    seldict['cpsr2'] = np.array([True if 'cpsr2' in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def system_cpsr2_20cm(backend_flags):
    seldict = dict()
    seldict['cpsr2_20cm'] = np.array([True if 'cpsr2_20cm' in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def system_cpsr2_50cm(backend_flags):
    seldict = dict()
    seldict['cpsr2_50cm'] = np.array([True if 'cpsr2_50cm' in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def system_wbcorr(backend_flags):
    seldict = dict()
    seldict['wbcorr'] = np.array([True if 'wbcorr' in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def custom_selection(flags,custom_flag_bor,flagval_bor):
    custom_flag = custom_flag_bor
    flagval = flagval_bor
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag_bor]==flagval_bor
    return seldict

def custom_selection_1(flags,custom_flag_bor_1,flagval_bor_1):
    custom_flag = custom_flag_bor_1
    flagval = flagval_bor_1
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_2(flags,custom_flag_bor_2,flagval_bor_2):
    custom_flag = custom_flag_bor_2
    flagval = flagval_bor_2
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_3(flags,custom_flag_bor_3,flagval_bor_3):
    custom_flag = custom_flag_bor_3
    flagval = flagval_bor_3
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_4(flags,custom_flag_bor_4,flagval_bor_4):
    custom_flag = custom_flag_bor_4
    flagval = flagval_bor_4
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_5(flags,custom_flag_bor_5,flagval_bor_5):
    custom_flag = custom_flag_bor_5
    flagval = flagval_bor_5
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_6(flags,custom_flag_bor_6,flagval_bor_6):
    custom_flag = custom_flag_bor_6
    flagval = flagval_bor_6
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_7(flags,custom_flag_bor_7,flagval_bor_7):
    custom_flag = custom_flag_bor_7
    flagval = flagval_bor_7
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_8(flags,custom_flag_bor_8,flagval_bor_8):
    custom_flag = custom_flag_bor_8
    flagval = flagval_bor_8
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_9(flags,custom_flag_bor_9,flagval_bor_9):
    custom_flag = custom_flag_bor_9
    flagval = flagval_bor_9
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_10(flags,custom_flag_bor_10,flagval_bor_10):
    custom_flag = custom_flag_bor_10
    flagval = flagval_bor_10
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_11(flags,custom_flag_bor_11,flagval_bor_11):
    custom_flag = custom_flag_bor_11
    flagval = flagval_bor_11
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_12(flags,custom_flag_bor_12,flagval_bor_12):
    custom_flag = custom_flag_bor_12
    flagval = flagval_bor_12
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_13(flags,custom_flag_bor_13,flagval_bor_13):
    custom_flag = custom_flag_bor_13
    flagval = flagval_bor_13
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_14(flags,custom_flag_bor_14,flagval_bor_14):
    custom_flag = custom_flag_bor_14
    flagval = flagval_bor_14
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_15(flags,custom_flag_bor_15,flagval_bor_15):
    custom_flag = custom_flag_bor_15
    flagval = flagval_bor_15
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval
    return seldict

def custom_selection_multi(flags,toas,custom_flag_bor_m,flagval_bor_m):
    seldict = dict()
    mask = np.repeat(True,len(toas))
    for cf, cv in zip(custom_flag_bor_m,flagval_bor_m):
      mask = mask * (flags[cf]==cv)
    seldict['_'.join(flagval_bor_m)] = mask
    return seldict

def custom_selection_split_1(freqs, flags,custom_flag_bor_s_1,flagval_bor_s_1,
                             custom_greater_bor_s_1,custom_freq_bor_s_1):
    custom_flag = custom_flag_bor_s_1
    flagval = flagval_bor_s_1
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError

    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval

    if custom_greater_bor_s_1:
      freq_mask = freqs >= custom_freq_bor_s_1
    else:
      freq_mask = freqs < custom_freq_bor_s_1

    seldict[flagval] = seldict[flagval] * freq_mask

    return seldict

def custom_selection_split_2(freqs, flags,custom_flag_bor_s_2,flagval_bor_s_2,
                             custom_greater_bor_s_2,custom_freq_bor_s_2):
    custom_flag = custom_flag_bor_s_2
    flagval = flagval_bor_s_2
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError

    seldict = dict()
    seldict[flagval] = flags[custom_flag]==flagval

    if custom_greater_bor_s_2:
      freq_mask = freqs >= custom_freq_bor_s_2
    else:
      freq_mask = freqs < custom_freq_bor_s_2

    seldict[flagval] = seldict[flagval] * freq_mask

    return seldict

def custom_selection_split_mid(freqs, flags, custom_flag_bor_sm,
                               flagval_bor_sm, custom_min_freq_bor_sm,
                               custom_max_freq_bor_sm):

    seldict = dict()
    seldict[flagval_bor_sm] = flags[custom_flag_bor_sm]==flagval_bor_sm

    freq_mask_greater = freqs >= custom_min_freq_bor_sm
    freq_mask_less = freqs < custom_max_freq_bor_sm

    seldict[flagval_bor_sm] = seldict[flagval_bor_sm] * freq_mask_greater * \
                                                 freq_mask_less

    return seldict

def system_jflag_pdfb4_20cm(flags):
    seldict = dict()
    seldict['pdfb4_jflag_20cm'] = np.array([True if 'pdfb4' in bb.lower() \
                                 and '20cm' in bb.lower() \
                                 else False for bb in flags['j']])
    return seldict

def system_pdfb1_early(backend_flags):
    seldict = dict()
    seldict['pdfb1_early'] = np.array([True if 'pdfb1_early' in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def system_pdfb1(backend_flags):
    seldict = dict()
    seldict['pdfb1'] = np.array([True if 'pdfb1' in bb.lower() \
                                 and 'pdfb_' not in bb.lower() \
                                 and 'pdfb1_early' not in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def system_pdfb(backend_flags):
    seldict = dict()
    seldict['pdfb'] = np.array([True if 'pdfb' in bb.lower() \
                                 and 'pdfb1' not in bb.lower() \
                                 else False for bb in backend_flags])
    return seldict

def ipta_J0437(freqs,backend_flags):
    """Selection function to split by backend flags."""
    seldict = dict()
    seldict['cpsr2_1400MHz'] = backend_flags == 'PKS.CPSR2.20CM'
    seldict['cpsr2_1400MHz_leg'] = backend_flags == 'PKS.cpsr2.20cm_legacy'
    seldict['1_2_GHz_rest'] = (([(freqs >= 1000)*(freqs < 2000)]) * (backend_flags != 'PKS.CPSR2.20CM') * (backend_flags != 'PKS.cpsr2.20cm_legacy'))[0,:]
    seldict['0_1_GHz'] = freqs < 1000
    seldict['2_inf_GHz'] = freqs >= 2000
    return seldict

def ipta_J1600(freqs,backend_flags):
    """Selection function to split by backend flags."""
    seldict = dict()
    seldict['Nancay_1400MHz'] = backend_flags == 'NRT.BON.1400'
    seldict['0_730_MHz'] = freqs <= 730
    return seldict

def ipta_J1643(freqs,backend_flags,flags):
    """Selection function to split by backend flags."""
    seldict = dict()
    seldict['Nancay_1400MHz'] = backend_flags == 'NRT.BON.1400'
    seldict['0_730_MHz'] = freqs <= 730
    seldict['750_890_MHz'] = (freqs >= 750)*(freqs <= 890)
    return seldict

def ipta_J1939(freqs,backend_flags):
    """Selection function to split by backend flags."""
    seldict = dict()
    seldict['Nancay_1400MHz'] = backend_flags == 'NRT.BON.1400'
    seldict['cpsr2_1400MHz'] = backend_flags == 'PKS.CPSR2.20CM'
    seldict['0_800_MHz'] = freqs <= 800
    seldict['2000_2500_MHz'] = (freqs >= 2000)*(freqs <= 2500)
    return seldict

def ipta_Nancay_1400MHz(freqs,backend_flags):
    """Selection function to split by backend flags."""
    seldict = dict()
    seldict['Nancay_1400MHz'] = backend_flags == 'NRT.BON.1400'
    return seldict

def single_band(freqs, min_freq=0, max_freq=10000, name='band'):
    """Selection only observations with min_freq <= f < max_freq"""
    return dict(zip([name], [(freqs >= min_freq)*(freqs < max_freq)]))


def nanograv_backends(backend_flags):
    """Selection function to split by NANOGRav backend flags only."""
    flagvals = np.unique(backend_flags)
    ngb = ["ASP", "GASP", "GUPPI", "PUPPI"]
    flagvals = filter(lambda x: any(map(lambda y: y in x, ngb)), flagvals)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def no_selection(toas):
    """Default selection with no splitting."""
    return {"": np.ones_like(toas, dtype=bool)}
