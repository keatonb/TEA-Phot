#!/usr/bin/env python

"""
PROGRAMME NAME:
    AEAP_beta.py

    Copyright (C) 2018
    D. M. Bowman (IvS, KU Leuven, Belgium)
    D. L. Holdsworth (JHI, UCLan, UK)

    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but without any warranty; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

DESCRIPTION:
    - Extract a differential light curve using adaptive elliptical apertures
        for FITS files.
    - Source extraction and aperature photometry is performed using the python
        SEP module: https://sep.readthedocs.io/en/v1.0.x/
    - The code is optimised for SHOC@SAAO photometry, but can and should be
        easily modified to work with other CCD instruments and telescopes.
    - BE AWARE: different instruments have different FITS headers and this
        requires a modification of the code.
    - This code requires user input at the command line and clicking on a plot,
        which is preceded by ">>>" in the terminal.
    - Warnings and errors are output to the terminal and are preceded by
        '! WARNING:' and '!!! ERROR:', respectively.

USAGE:
    - For help, type at the command line: python AEAP_beta.py --help
    - MINIMUM USAGE:
        python AEAP_beta.py <observatory> <instrument> <image.fits>

    - OPTIONS:
        python AEAP_beta.py <observatory> <instrument> <image.fits> --bias <bias.fits> --dark <dark.fits> --flat <flat.fits> --object <object> --coord <"19:22:43.56 -21:25:27.53"> --source_sigma <10> --extract <True> --do_plot <True> --do_clip <10> --extinction <0.25> --image_dir <./> --bias_dir <./> --flat_dir <./> --out_dir <./>

WARNINGS WHILST RUNNING (SUPPRESSED):
    - WARNING: VerifyWarning: Card is too long, comment will be truncated.
        [astropy.io.fits.card]

KNOWN ISSUES/BUGS:
    - programme crahses if multiple (red) sources are within pixel limit of
        user's choice. This usually only occurs if the sources are blurred and
        the resultant PSFs are highly elliptical such that sep.extract() finds
        multiple ellipses at a single star's location
    - if user closes a window, programme will hang and becomes difficult to
        kill from the terminal
    - sep.sum_ellipse() returns ZeroDivisionError if annuli option is chosen
        and they are larger than the CCD image frame

"""

from __future__ import division

import sys
import os
import argparse
import numpy as np
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep as sep

from astropy.io import fits
from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord, AltAz

import warnings
warnings.simplefilter("ignore")


###############################################################################
# FUNCTION: bias, dark and flat field correction

def image_reduction(instrument, bias, bias_dir, flat, flat_dir, do_plot,
                    out_dir, **kwargs):

    # SHOC INSTRUMENT...
    if instrument == "SHOC":

        # BIASES: combine and take median of BIAS frames
        bias_filename, bias_ext = os.path.splitext(bias)
        if bias_ext == ".fz":
            bias_filename, bias_ext = os.path.splitext(bias_filename)
            bias_ext = ".fits.fz"

        bias_hdulist = fits.open(bias_dir+'/'+bias, ignore_missing_end=True)
        if bias_ext == ".fits":
            bias_hdu = bias_hdulist[0]
        if bias_ext == ".fits.fz":
            bias_hdu = bias_hdulist[1]

        n_bias_slice = bias_hdu.data.shape[0]
        if n_bias_slice > 1:
            master_bias = np.median(bias_hdu.data, axis=0)
        else:
            print("\n     ! WARNING: only 1 bias frame detected \n")

        """
        # DARKS: combine and take median of DARK frames
        Note that in professional observatories the CCDs are cooled so the dark
        count is constant so it is typical to only perform a bias subtraction
        """

        # FLAT: combine and normalise by mean of FLAT frames
        flat_filename, flat_ext = os.path.splitext(flat)
        if flat_ext == ".fz":
            flat_filename, flat_ext = os.path.splitext(flat_filename)
            flat_ext = ".fits.fz"

        flat_hdulist = fits.open(flat_dir+'/'+flat, ignore_missing_end=True)
        if flat_ext == ".fits":
            flat_hdu = flat_hdulist[0]
        if flat_ext == ".fits.fz":
            flat_hdu = flat_hdulist[1]
        n_flat_slice = flat_hdu.data.shape[0]
        x_flat_size = flat_hdu.data.shape[1]
        y_flat_size = flat_hdu.data.shape[2]

        if n_flat_slice > 1:
            norm_flat = []
            for x in range(n_flat_slice):
                flat = flat_hdu.data[x, :, :].astype(dtype=float)
                flat_mean = np.mean(flat_hdu.data[x, :, :].astype(dtype=float))
                norm_flat.append((flat / flat_mean))

            master_flat = np.median(norm_flat, axis=0)
        else:
            print("\n     ! WARNING: only 1 flat frame detected \n")

    # STE INSTRUMENT...
    elif instrument == "STE":

        # BIASES: combine and take median of BIAS frames
        bias_files = []
        bias_concat = []
        for file in os.listdir(bias_dir):
            if bias in file:
                bias_files.append(file)

        if len(bias_files) == 0:
            print("     !!! ERROR: no bias files containing the string: "+str(bias)+" found in:")
            print("               "+str(bias_dir))
            exit()

        # read if BIAS file extension is .fits OR .fits.fz
        bias_filename, bias_ext = os.path.splitext(bias_files[0])
        if bias_ext == ".fz":
            bias_filename, bias_ext = os.path.splitext(bias_filename)
            bias_ext = ".fits.fz"

        for i in bias_files:
            bias_hdulist = fits.open(bias_dir+'/'+i, ignore_missing_end=True)
            if bias_ext == ".fits":
                bias_hdu = bias_hdulist[0]
            if bias_ext == ".fits.fz":
                bias_hdu = bias_hdulist[1]

            # check if FITS header lists each file as a BIAS
            bias_header = bias_hdu.header
            if bias_header['OBJECT'] not in ["BIAS"]:
                print("     ! WARNING: are you sure that this is a BIAS: "+str(i)+"\n")
                continue

            else:
                # crop BIAS to use only useful region
                trim = header['TRIMSEC']  # "useful part of the data"
                new_trim = re.findall(r'\d+', trim)
                trim_x1, trim_x2, trim_y1, trim_y2 = new_trim[0], new_trim[1], new_trim[2], new_trim[3]
                # s = header['BIASSEC'] # "bias overscan region"
                bias_data = bias_hdu.data.astype(np.float)
                # bias_data = bias_data[int(trim_y1):int(trim_y2), int(trim_x1):int(trim_x2)]  # removed
                bias_concat.append(bias_data)

        # take median of all biases to make master bias
        master_bias = np.median(bias_concat, axis=0)

        """
        # DARKS: combine and take median of DARK frames
        """

        # FLATS: combine and normalise by mean of FLAT frames
        flat_files = []
        flat_concat = []
        for file in os.listdir(flat_dir):
            if flat in file:
                flat_files.append(file)

        if len(flat_files) == 0:
            print("     !!! ERROR: no flat files containing the string: "+str(flat)+" found in:")
            print("               "+str(flat_dir))
            exit()

        # read if FLAT file extension is .fits OR .fits.fz
        flat_filename, flat_ext = os.path.splitext(flat_files[0])
        if flat_ext == ".fz":
            flat_filename, flat_ext = os.path.splitext(flat_filename)
            flat_ext = ".fits.fz"

        norm_flat = []
        for i in flat_files:
            flat_hdulist = fits.open(flat_dir+'/'+i, ignore_missing_end=True)
            if flat_ext == ".fits":
                flat_hdu = flat_hdulist[0]
            if flat_ext == ".fits.fz":
                flat_hdu = flat_hdulist[1]

            # check if FITS header lists each file as a FLAT
            flat_header = flat_hdu.header
            if flat_header['OBJECT'] not in ["SKYFLAT"]:
                print("     ! WARNING: are you sure that this is a FLAT: "+str(i)+"\n")
                continue

            else:
                # crop FLAT to use only useful region
                trim = header['TRIMSEC']  # "useful part of the data"
                new_trim = re.findall(r'\d+', trim)
                trim_x1, trim_x2, trim_y1, trim_y2 = new_trim[0], new_trim[1], new_trim[2], new_trim[3]
                # s = header['BIASSEC'] # "bias overscan region"
                flat_data = flat_hdu.data.astype(np.float)
                # flat_data = flat_data[int(trim_y1):int(trim_y2), int(trim_x1):int(trim_x2)]  # removed
                flat_mean = np.mean(flat_data)
                norm_flat.append((flat_data / flat_mean))

        # take median of all flats to make a master flat
        master_flat = np.median(norm_flat, axis=0)

    else:
        print("\n     !!! ERROR: unknown instrument specified")
        print("                Exiting program \n")
        exit()

    # plotting of master files
    if do_plot:
        print("\n (*) Plotting master BIAS frame ...")
        fig1, ax = plt.subplots(figsize=(6, 4.5), num=1)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
        m, s = np.mean(master_bias), np.std(master_bias)
        im = ax.imshow(master_bias, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')
        tick_plot(ax, master_bias[0], master_bias[1])  # call tick function
        plt.colorbar(im, label="Counts", extend='both')
        plt.savefig(out_dir+"master_bias.eps", dpi=300)
        plt.close(1)

    if do_plot:
        print("\n (*) Plotting master FLAT frame ...")
        fig2, ax = plt.subplots(figsize=(6, 4.5), num=2)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
        m, s = np.mean(master_flat), np.std(master_flat)
        im = ax.imshow(master_flat, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')
        tick_plot(ax, master_flat[0], master_flat[1])  # call tick function
        plt.colorbar(im, label="Normalised Counts", extend='both')
        plt.savefig(out_dir+"master_flat.eps", dpi=300)
        plt.close(2)

    return master_bias, master_flat


###############################################################################
# FUNCTION: ellipse plotting on CCD frames...

def ellipse_plot(x, y, size, a, b, theta, f_colour, e_colour, number):

    if len(number) != 1:
        for i in number:
            e = Ellipse(xy=(x, y), width=2.0*i*size*a, height=2.0*i*size*b,
                        angle=theta * 180.0 / np.pi)
            e.set_facecolor(f_colour)
            e.set_edgecolor(e_colour)
            if i > 1:
                e.set_linestyle('--')
            ax.add_artist(e)
            plt.draw()
    else:
        e = Ellipse(xy=(x, y), width=2.0*size*a, height=2.0*size*b,
                    angle=theta * 180.0 / np.pi)
        e.set_facecolor(f_colour)
        e.set_edgecolor(e_colour)
        ax.add_artist(e)
        plt.draw()


###############################################################################
# FUNCTION: add major and minor tick marks to a plot...

def tick_plot(ax, x, y):
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_xticks(np.arange(0, len(x), 50))
    ax.set_yticks(np.arange(0, len(y), 50))
    ax.set_xticks(np.arange(0, len(x), 10), minor=True)
    ax.set_yticks(np.arange(0, len(y), 10), minor=True)


###############################################################################
# FUNCTION: find and return observatory coordinates (lon, lat, alt)

def get_observatory(observatory, **kwargs):
    """
    If you want a *precise* location of a telescope, you can define it below:
    """
    if observatory == "SAAO":
        obs_loc = coord.EarthLocation.from_geodetic(lat=-32.379663,
                                                    lon=20.810129,
                                                    height=1798.0)
    else:
        try:
            obs_loc = coord.EarthLocation.of_site(observatory).to_geodetic()
        except:
            print("\n     !!! ERROR: that observatory is not known by default. You can enter the location manually in the source code, or choose one of the following:")
            print("\n    ('%s')" % "','".join(coord.EarthLocation.get_site_names())[9:]+"\n")
            exit()

    return obs_loc


###############################################################################
# FUNCTION: find exposure time and midpoint from FITS files

def find_midpoint(instrument, header, iter):

    # SHOC...
    if instrument == "SHOC":

        trigger = np.str(header['TRIGGER'])  # needed to define exposure time

        # Internal trigger: FRAME timestamp is *END* of the first exposure
        # Note: that the FITS file comment is WRONG (see SHOC paper)
        if trigger == "Internal":
            exp_time = np.float(header['Exposure'])      # units of sec
            ACT_time = np.float(header['ACT'])           # units of sec
            read_time = (ACT_time - exp_time) / 86400.0  # units of days
            exp_time = exp_time / 86400.0                # convert into days

            # read time_stamp and *SUBTRACT* half exp_time to get midpoint
            t0 = time.Time(header['FRAME'], format='isot', scale='utc',
                           precision=9)
            midpoint = t0.utc.jd - (0.5*exp_time)

        # External Start: FRAME timestamp is *END* of the first exposure
        # Note: that the FITS file comment is WRONG (see SHOC paper)
        elif trigger == "External Start":
            exp_time = np.float(header['Exposure'])     # units of sec
            exp_time = exp_time / 86400.0               # convert into days
            read_time = 0.00676 / 86400.0               # hard-coded as not in FITS header

            # read time_stamp and *SUBTRACT* half exp_time to get midpoint
            t0 = time.Time(header['FRAME'], format='isot', scale='utc',
                           precision=9)
            midpoint = t0.utc.jd - (0.5*exp_time)

        # External trigger: FRAME timestamp is *START* of the first exposure
        elif trigger == "External":
            exp_time = np.float(header['GPS-INT'])      # units of msec
            exp_time = exp_time / 1000.0 / 86400.0      # convert into days
            read_time = 0.00676 / 86400.0               # hard-coded as not in FITS header

            # read time_stamp and *ADD* half exp_time to get midpoint
            t0 = time.Time(header['FRAME'], format='isot', scale='utc',
                           precision=9)
            midpoint = t0.utc.jd + (0.5*exp_time)

        else:
            print("\n     !!! ERROR: 'TRIGGER' keyword not found in FITS header")
            print("                Are you sure this is a SHOC FITS file? \n")
            exit()

        # populate JD midpoint time stamp of 'ith' image frame
        increment = ((float(iter)-2.0)*(read_time + exp_time))
        midpoint_JD = (midpoint + increment)

    # STE...
    elif instrument == "STE":

        # read integration (exp_time+read_time) from FITS header
        exp_time = np.float(header['EXPTIME'])  # units of sec
        exp_time = exp_time / 86400.0           # units of days

        # STE gives UT stamps at *START* of exposures so add on half exp time
        t0_date = header['DATE-OBS']
        t0_time = header['UT']
        obs_date = t0_date+"T"+t0_time
        t0 = time.Time(obs_date, format='isot', scale='ut1', precision=9)
        midpoint_JD = (t0.ut1.jd + 0.5*exp_time)

        read_time = 0.0  # since time stamp is accurate for each exposure
        trigger = ""

    else:
        print("\n     !!! ERROR: unknown instrument specified")
        print("                Exiting program \n")
        exit()

    return exp_time, read_time, t0, midpoint_JD, trigger


###############################################################################
# FUNCTION: find gain sensitivity gain (electrons per ADU) for instrument

def find_instrument(instrument, header):

    # SHOC: coded such that outptamp == "conventional"
    if instrument == "SHOC":

        # read specific keywords from FITS header
        serial_num = np.str(header['SERNO'])        # instrument serial number
        preamp_gain = np.str(header['PREAMP'])      # preamp gain used to look-up sensitivity gain
        pixelreadtime = np.str(header['READTIME'])  # pixel readtime
        outptamp = np.str(header['OUTPTAMP'])       # typically set as conventional

        # SHOC1 (SHA)
        if serial_num == "5982":
            if pixelreadtime == "3.333333e-07":
                sensitivity = {'1.0': 10.98, '2.4': 4.23, '4.9': 1.82}
                gain = sensitivity[preamp_gain]
            if pixelreadtime == "1e-06":
                sensitivity = {'1.0': 4.06, '2.4': 1.69, '4.9': 0.63}
                gain = sensitivity[preamp_gain]

        # SHOC2 (SHD)
        elif serial_num == "6448":
            if pixelreadtime == "3.333333e-07":
                sensitivity = {'1.0': 9.92, '2.5': 3.96, '5.2': 1.77}
                gain = sensitivity[preamp_gain]
            if pixelreadtime == "1e-06":
                sensitivity = {'1.0': 3.79, '2.5': 1.53, '5.2': 0.68}
                gain = sensitivity[preamp_gain]

        else:
            print("\n     !!! ERROR: unknown 'SERIAL_NUM' in FITS header")
            print("                Are you sure this is a SHOC FITS file? \n")
            exit()

    # STE...
    elif instrument == "STE":
        gain = np.float(header['GAIN'])

    else:
        print("\n     !!! ERROR: unknown instrument specified")
        print("                Exiting program \n")
        exit()

    return gain


###############################################################################
# FUNCTION: find object name in FITS header for a given instrument

def find_object_name(instrument, header):

    # NOTE: only coded for SHOC
    if instrument == "SHOC":
        star_name = np.str(header['OBJECT'])

    elif instrument == "STE":
        star_name = np.str(header['OBJECT'])

    else:
        print("\n     !!! ERROR: unknown instrument specified")
        print("                Exiting program \n")
        exit()

    return star_name


###############################################################################
# FUNCTION: find object coordinates in FITS header for a given instrument

def find_object_coord(instrument, header):

    # NOTE: only coded for SHOC
    if instrument == "SHOC":
        star_ra = np.str(header['OBJRA'])
        star_dec = np.str(header['OBJDEC'])
    elif instrument == "STE":
        star_ra = np.str(header['RA'])
        star_dec = np.str(header['DEC'])
    else:
        print("\n     !!! ERROR: unknown instrument specified")
        print("                Exiting program \n")
        exit()

    return star_ra, star_dec


###############################################################################
# FUNCTION: write FITS files for reduced images and *corrected* FITS headers

def write_new_fits(instrument, hdu, red_image, fit_path, star_name,
                   image_filename, label, t0, JD_time, midpoint_JD,
                   midpoint_HJD, midpoint_BJD, read_time, airmass):

    # NOTE: only coded for SHOC
    if instrument == "SHOC":
        new_hdu = fits.PrimaryHDU()
        new_hdu.header['COMMENT'] = ("  FITS (Flexible Image Transport System) format is defined in 'Astronomy")
        new_hdu.header['COMMENT'] = ("  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H")
        new_hdu.header['DATATYPE'] = ('counts',"Data type")
        new_hdu.header['IMGRECT'] = (str(hdu.header['IMGRECT']),"Image format")
        new_hdu.header['HBIN'] = (str(hdu.header['hbin']),"Horizontal binning")
        new_hdu.header['VBIN'] = (str(hdu.header['vbin']),"Vertical binning")
        if instrument == "SHOC":
            if trigger == "Internal":
                new_hdu.header['EXPOSURE'] = (str(hdu.header['EXPOSURE']),"exposure time (sec)")
                new_hdu.header['ACT'] = (str(hdu.header['ACT']),"integration cycle time (sec)")
            elif trigger == "External Start":
                new_hdu.header['EXPOSURE'] = (str(hdu.header['EXPOSURE']),"exposure time (sec)")
                new_hdu.header['READTIME'] = (str(read_time*86400.0),"CCD read out time (sec)")
            elif trigger == "External":
                new_hdu.header['GPSSTART'] = (str(hdu.header['GPSSTART']),"GPS start time (UTC; external)")
                new_hdu.header['GPS-INT'] = (str(hdu.header['GPS-INT']),"GPS trigger interval (msec)")
        new_hdu.header['TRIGGER'] = (str(hdu.header['TRIGGER']),"Trigger mode")
        new_hdu.header['READTIME'] = (str(hdu.header['READTIME']),"Pixel read time")
        new_hdu.header['OUTPTAMP'] = (str(hdu.header['OUTPTAMP']),"Output Amplifier")
        new_hdu.header['PREAMP'] = (str(hdu.header['PREAMP']),"Pre Amplifier Gain")
        new_hdu.header['DATE'] = (str(t0),"file creation date (YYYY-MM-DDThh:mm:ss)")
        new_hdu.header['FRAME'] = (str(JD_time.utc.isot),"Midpoint of frame exposure")
        new_hdu.header['DATE-OBS'] = (str(hdu.header['DATE-OBS']),"The time the user pushed start (UTC)")
        new_hdu.header['AIRMASS'] = (str(airmass),"The airmass (sec(z))")
        new_hdu.header['FILTERA'] = (str(hdu.header['FILTERA']),"the active filter in wheel A")
        new_hdu.header['FILTERB'] = (str(hdu.header['FILTERB']),"the active filter in wheel B")
        new_hdu.header['SERNO'] = (str(hdu.header['SERNO']),"The SHOC camera serial number")
        new_hdu.header['JD-MID'] = (str(midpoint_JD-2450000.0),"midpoint of exposure (JD-2450000.0)")
        new_hdu.header['HJD-MID'] = (str(midpoint_HJD),"midpoint of exposure (HJD-2450000.0 UTC)")
        new_hdu.header['BJD-MID'] = (str(midpoint_BJD),"midpoint of exposure (BJD-2450000.0 TDB)")

        # write out new image FITS files with zero padding in suffix number
        new_hdu.data = np.array((red_image))
        new_hdu.writeto(fit_path+star_name+"_"+image_filename+"_"+label+".fits",
                        overwrite=True)

    elif instrument == "STE":
        print("No info for STE in write_new_fits")

    else:
        print("\n     !!! ERROR: unknown instrument specified")
        print("                Exiting program \n")
        exit()


###############################################################################
# FUNCTION: handling of command line arguments (e.g. str, bool, etc)

def two_strs(v):
    values = v.split()
    if len(values) != 2:
        raise argparse.ArgumentTypeError('\n\n !!! ERROR: RA and DEC coordinates need to be provided in the form: \n            --coord "19:22:43.56 -21:25:27.53" \n')
    values = list(map(str, values))
    return values


def str2bool(v):
    if v.lower() in ('yes',  'y', 'true', 't', 'T', 'True', 'TRUE', '1'):
        return True
    elif v.lower() in ('no', 'n', 'false', 'f', 'False', 'FALSE', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('\n\n !!! ERROR: a boolean value is expected for these optional parameters: \n            --do_plot True \n            --extract True')


def str2flt(v):
    if v.replace('.', '', 1).isdigit():
        return float(v)
    else:
        raise argparse.ArgumentTypeError('\n\n !!! ERROR: a float is expected for these optional parameters: \n            --source_sigma 10 \n            --do_clip 5 \n            --extinction 0.25 \n')


###############################################################################
# # # # # # # # # # # # # # BELOW IS THE MAIN CODE # # # # # # # # # # # # # #
###############################################################################

if __name__ == '__main__':

    # set default plotting parameters
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 14}

    matplotlib.rc('font', **font)
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    # clear the terminal
    os.system("clear")

    # read command line arguments
    parser = argparse.ArgumentParser(prog="AEAP_beta.py",
                                     description="------------ Adaptive Elliptical Aperture Photometry ------------",
                                     epilog='''
                                            AEAP_beta.py  Copyright (C) 2018
                                            D. M. Bowman (KU Leuven) and D. L. Holdsworth (UCLan).
                                            This program comes with ABSOLUTELY NO WARRANTY.
                                            This is free software, and you are welcome to redistribute it.
                                            ''')
    parser.add_argument('observatory', type=str, help="Observatory name (e.g. SAAO); <str>")
    parser.add_argument('instrument', type=str, help="Instrument name (e.g. SHOC or STE); <str>")
    parser.add_argument('image', type=str, help="If using the SHOC instrument <image> is the filename of image FITS cube; if using the STE instrument <image> is automatically set to 'a' <str>")

    parser.add_argument('--bias', type=str, help="Filename of bias FITS cube (with SHOC), or a common pre-fix string to search for multiple bias files (e.g. STE); <str>")
    #parser.add_argument('--dark', type=str, help="Filename of dark FITS cube (with SHOC), or a or a common pre-fix string to search for multiple dark files (e.g. STE); <str>") # TO DO
    parser.add_argument('--flat', type=str, help="Filename of flat FITS cube (with SHOC), or a or a common pre-fix string to search for multiple flat files (e.g. STE); <str>")

    parser.add_argument('--image_dir', type=str, default='./', action='store', help="Directory path of where image file(s) are located; <str>")
    parser.add_argument('--bias_dir', type=str, default='./', action='store', help="Directory path of where bias file(s) are located; <str>")
    #parser.add_argument('--dark_dir', type=str, default='./', action = 'store', help="Directory path of where dark file(s) are located; <str>") # TO DO
    parser.add_argument('--flat_dir', type=str, default='./', action='store', help="Directory path of where flat file(s) are located; <str>")
    parser.add_argument('--out_dir', type=str, default='./', action='store', help="Directory path of where code output; <str>")
    parser.add_argument('--object', default=None, type=str, help="Overwrite the object name in FITS header; <str>")
    parser.add_argument('--coord', default=None, type=two_strs, help="Overwrite the coordinates [ra,dec] of target star in FITS header; <'00:00:00.00 -00:00:00.00'> ")
    parser.add_argument('--source_sigma', default='10', action='store', type=str2flt, help="The S/N of the image source extraction; <float>")
    parser.add_argument('--do_plot', default=False, type=str2bool, help="Plot and save all figures; <bool>")
    parser.add_argument('--extract', default=False, type=str2bool, help="Extract and save individual FITS files containing reduced images in a subdirectory; <bool>")
    parser.add_argument('--do_clip', default=None, type=str2flt, help="Sigma value for outlier removal in differential light curve; <float>")
    parser.add_argument('--extinction', default=None, type=str2flt, help="Include extinction correction to individual light curves (units of magnitude); <float>")

    # EXAMPLE: parser.add_argument('--xlim', help = 'X axis limits', action = 'store', type=two_floats, default = [-1.e-3, 1.e-3])
    args = parser.parse_args()

    # define command line arguments
    observatory = args.observatory
    instrument = args.instrument
    image = args.image
    source_sigma = args.source_sigma
    do_plot = args.do_plot
    do_clip = args.do_clip
    ext_coeff = args.extinction
    extract = args.extract

    # define command line prompt options
    Y_proceed = ["y", "Y", "yes", "Yes", "YES"]
    N_proceed = ["n", "N", "no", "No", "NO"]
    Q_proceed = ["Quit", "QUIT", "quit", "Exit", "exit", "EXIT"]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 'changeable inputs' for APERTURE PHOTOMETRY (only for the expert)

    # size of plotting ellipses in images: not used for science
    size = 4

    # sizes used for curve of growth assessment for optimum aperture size
    ap_size = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # pixel tolerance for jitter in source extraction from frame-to-frame
    pix_limit = 10

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # get coordinates (lon, lat, alt) of specified observatory
    obs_loc = get_observatory(observatory)

    # check if user-provided instrument is known
    """
    # TIP: you will need to include alternative observatory and instrument
    names below, but also at all other necessary locations (and functions) in
    the code
    """
    known_instruments = ["SHOC", "STE"]

    if instrument == "SHOC":
        cube = True
    elif instrument == "STE":
        """
        # In the case of STE, the user's input is over-written since all
        STE FITS files always start with 'a'. The code *requires* FITS files
        to be orgainised into approproprite directories
        """
        image, bias, flat = "a", "a", "a"  # STE FITS files all start with 'a'
        cube = False
        extract = False  # only allow extract if cube not False
    else:
        if instrument not in known_instruments:
            print("\n\n     !!! ERROR: unknown instrument")
            print("\n               Available instrument keywords are:")
            print("               "+str(known_instruments)+"\n")
            exit()

    # define empty lists for light curves
    final_time_HJD = []
    final_time_BJD = []
    targ_mag = []
    targ_mag_err = []
    comp_mag = []
    comp_mag_err = []
    targ_mag_corr = []
    comp_mag_corr = []
    targ_airmass = []

    # define empty lists for curve of growth
    cg_area = []
    cg_flux = []
    cg_sky = []
    cg_area_comp = []
    cg_flux_comp = []

    # check that file directories exist
    image_dir = args.image_dir
    if not os.path.exists(image_dir):
        print("\n     !!! ERROR: specified IMAGE directory does not exist:")
        print("               "+str(image_dir)+"\n")
        exit()
    bias_dir = args.bias_dir
    if not os.path.exists(bias_dir):
        print("\n     !!! ERROR: specified BIAS directory does not exist:")
        print("               "+str(bias_dir)+"\n")
        exit()
    """
    dark_dir = args.dark_dir
    """
    flat_dir = args.flat_dir
    if not os.path.exists(flat_dir):
        print("\n     !!! ERROR: specified FLAT directory does not exist:")
        print("               "+str(flat_dir)+"\n")
        exit()
    out_dir = args.out_dir
    out_dir = os.path.dirname(out_dir)
    if not os.path.exists(out_dir):
        print("\n (*) The specfied 'out_dir' did not exist, so it has been created \n")
        os.makedirs(out_dir)
    out_dir = out_dir+"/"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # BELOW SETS UP TO CUBE MODULE (E.G. SHOC AT SAAO)...
    """
    SHOC photometry from SAAO is output as 3D FITS cubes, so the individual
    image, bias and flat frames need to be extracted before they can be reduced
    and aperture photometry done.
    """
    if cube:

        # get image file name
        if os.path.isfile(image_dir+'/'+image) is False:
            print("\n     !!! ERROR: specified IMAGE does not exist at:")
            print("               <"+str(image_dir+'/'+image)+"> \n")
            exit()

        # get bias name if provided
        if args.bias:
            bias = args.bias
            if os.path.isfile(bias_dir+'/'+bias) is False:
                print("\n     !!! ERROR: specified BIAS does not exist at:")
                print("               <"+str(bias_dir+'/'+bias)+"> \n")
                exit()

        """
        # get dark name if provided:
        if args.dark:
            dark = args.dark
            if os.path.isfile(dark_dir+'/'+dark) == False:
                print("\n     !!! ERROR: specified DARK does not exist at:")
                print("               <"+str(dark_dir+'/'+dark)+"> \n")
                exit()
        """

        # get flat name if provided
        if args.flat:
            flat = args.flat
            if os.path.isfile(flat_dir+'/'+flat) is False:
                print("\n     !!! ERROR: specified FLAT does not exist at:")
                print("               <"+str(flat_dir+'/'+flat)+"> \n")
                exit()

        # read if IMAGE cube file extension is .fits OR .fits.fz
        image_filename, image_ext = os.path.splitext(image)
        if image_ext == ".fz":
            image_filename, image_ext = os.path.splitext(image_filename)
            image_ext = ".fits.fz"

        # read image FITS CUBE file header and data
        hdulist = fits.open(image_dir+'/'+image, ignore_missing_end=True)
        if image_ext == ".fits":
            hdu = hdulist[0]
            header = hdulist[0].header
        if image_ext == ".fits.fz":
            hdu = hdulist[1]
            header = hdulist[1].header
        n_slice = hdu.data.shape[0]
        data = hdu.data

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # BELOW SETS UP TO NON-CUBE MODULE (E.G. STE AT SAAO)...

    if cube is False:

        # search for IMAGE files in image_dir
        image_files = []
        for file in os.listdir(image_dir):
            if image in file:
                image_files.append(file)

        if len(image_files) == 0:
            print("\n     !!! ERROR: no IMAGE files containing the string: <"+str(image)+"> found in: <"+str(image_dir)+"> directory \n")
            exit()
        n_slice = len(image_files)

        # read if IMAGE file extension is .fits OR .fits.fz
        image_filename, image_ext = os.path.splitext(image_files[0])
        if image_ext == ".fz":
            image_filename, image_ext = os.path.splitext(image_filename)
            image_ext = ".fits.fz"

        # read 'first' image FITS file header
        hdulist = fits.open(image_dir+'/'+image_files[0],
                            ignore_missing_end=True)
        if image_ext == ".fits":
            hdu = hdulist[0]
            header = hdulist[0].header
        if image_ext == ".fits.fz":
            hdu = hdulist[1]
            header = hdulist[1].header

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # BELOW SETS UP APERTURE PHOTOMETRY

    # print pre-amble to terminal
    print("\n (*) Running aperture photometry script for "+str(instrument)+" photometry from "+str(observatory)+" ...")

    # allow user to overwrite FITS file object name in header
    if args.object:
        star_name = args.object
        print("     > Using user-defined name of object: "+str(star_name))
    else:
        star_name = find_object_name(instrument, header)
        print("     > Using name of object from FITS header: "+str(star_name))

    # read RA and DEC from fits header if not provided at the command line
    if args.coord:
        star_ra = args.coord[0]
        star_dec = args.coord[1]
        print("     > Using user-defined RA and DEC coordinates: "+str(star_ra)+" "+str(star_dec))
    else:
        star_ra, star_dec = find_object_coord(instrument, header)
        print("     > Using the RA and DEC from FITS header: "+str(star_ra)+" "+str(star_dec))
        print("\n     ! WARNING: have you checked the that the contents of the FITS header(s) are correct?")
        print("     It is advised to enter the name and RA and DEC of your target manually at the command line using:")
        print("     --object <HD123456> --coord <'00:00:00 -00:00:00'> \n")

    # define S/N for source extraction from command line (default of 10)
    if args.source_sigma:
        print("     > Using source extraction S/N criterion of: "+str(source_sigma))

    # calculate bias and flat-field master frames
    if args.flat and args.bias:
        master_bias, master_flat = image_reduction(instrument, bias, bias_dir,
                                                   flat, flat_dir, do_plot,
                                                   out_dir)

    # create directory for extracted FITS files if it doesnt exist
    if extract:
        fit_path = out_dir+image_filename+"/"
        fit_dir = os.path.dirname(fit_path)
        if not os.path.exists(fit_dir):
            os.makedirs(fit_dir)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # BELOW IS THE LOOP OVER ALL IMAGE FRAMES...

    # define starting image number to loop over
    """
    Note: the first two image frames should always be thrown away when trigger
    is External or External Start when using SHOC. In other cases, the loop of
    image files starts at a counter of zero.
    """
    if instrument == "SHOC":
        n_start = 2
    elif instrument == "STE":
        n_start = 0
    else:
        n_start = 0

    # define looping criteria for images
    for i in range(n_start, n_slice):

        # boolean to do aperture photometry unless error
        do_targ, do_comp = True, True

        # find correct gain sensitivity from instrument look-up table
        gain = find_instrument(instrument, header)

        # find exposure time and image frame midpoint (in JD)
        exp_time, read_time, t0, midpoint_JD, trigger = find_midpoint(instrument, header, int(i))

        # define time of first exposure in cube
        if int(i) == n_start:
            t0_start = t0

        if i == n_start:
            if instrument == "SHOC":
                print("\n     > SHOC triggering is: "+str(trigger))
            print("     > Exposure time is: "+str(exp_time*86400.0)+" (sec)")

        # read individual 'ith' image frame (e.g. SHOC)
        if cube:
            hdu.data = data[i, :, :]
        # read image 'ith' FITS file header and data (e.g. STE)
        else:
            hdulist = fits.open(image_dir+'/'+image_files[i], ignore_missing_end=True)
            if image_ext == ".fits":
                hdu = hdulist[0]
                header = hdulist[0].header
            if image_ext == ".fits.fz":
                hdu = hdulist[1]
                header = hdulist[1].header
            hdu.data = hdu.data

        # Create an image mask to avoid overscan regions on edges of CCD
        if instrument == "STE":
            trim = header['TRIMSEC']  # "useful part of the data"
            new_trim = re.findall(r'\d+', trim)
            trim_x1, trim_x2, trim_y1, trim_y2 = new_trim[0], new_trim[1], new_trim[2], new_trim[3]
            # s = header['BIASSEC'] # "bias overscan region"
        else:
            trim_x1, trim_x2, trim_y1, trim_y2 = 0, hdu.data.shape[0], 0, hdu.data.shape[1]

        # reductions of raw to reduced image frames
        raw_image = hdu.data.astype(dtype=float)
        # raw_image = raw_image[int(trim_y1):int(trim_y2), int(trim_x1):int(trim_x2)]  # removed as not needed to crop image

        # create an image mask for source extraction
        # masks work in extract such that True pixels are not considered
        edge_mask = np.zeros(raw_image.shape, dtype=np.bool)
        edge_mask[:] = False
        if instrument == "STE":
            edge_mask[:40, :] = True
            edge_mask[:, :11] = True
            edge_mask[:, 263:] = True
            edge_mask[250:, :] = True

        # do image reduction if FLAT and BIAS arguments are provided
        if args.flat and args.bias:
            red_image = ((raw_image - master_bias)/master_flat)
        else:
            red_image = raw_image

        # create JD astropy time object
        JD_time = time.Time(midpoint_JD, format='jd', scale='utc', location=obs_loc)
        star_coord = SkyCoord(star_ra, star_dec, frame='icrs', unit=(u.hourangle, u.deg))

        # calculate HJD (UTC) and truncate
        ltt_helio = JD_time.light_travel_time(star_coord, 'heliocentric')
        midpoint_HJD = (JD_time.utc+ltt_helio)
        midpoint_HJD = np.float(midpoint_HJD.value)-2450000.0

        # calculate BJD (TDB) and truncate
        # Note: HJD (UTC) and BJD (TDB) can be up to +/- 4 mins different
        ltt_bary = JD_time.light_travel_time(star_coord, 'barycentric')
        midpoint_BJD = (JD_time.tdb+ltt_bary)
        midpoint_BJD = np.float(midpoint_BJD.value)-2450000.0

        # calculate airmass correction using target star's coordinates
        staraltaz = star_coord.transform_to(AltAz(obstime=JD_time, location=obs_loc))
        z = 90.0 - (staraltaz.alt.degree)
        z = np.deg2rad(z)
        secz = 1.0 / np.cos(z)
        X = secz - 0.0018167*(secz-1.0) - 0.002875*((secz-1.0)**2.0) - 0.0008083*((secz-1.0)**3.0)

        # create a new FITS file for each image frame for FITS cubes
        if extract:
            label = '{:04d}'.format(i)
            write_new_fits(instrument, hdu, red_image, fit_path, star_name,
                           image_filename, label, t0, JD_time, midpoint_JD,
                           midpoint_HJD, midpoint_BJD, read_time, X)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # BELOW DOES THE IMAGE FRAME REDUCTION AND USER SOURCE SELECTION...

        # subtract spatially varying background: needed for sep.extract()
        bkg = sep.Background(red_image, mask=edge_mask)
        red_image = red_image - bkg

        # SEP source extraction requiring minimum of 8 pixels for each source
        objects = sep.extract(red_image, source_sigma, err=bkg.globalrms,
                              mask=edge_mask, minarea=8, gain=gain)

        # define min/max location of all extracted sources
        x_min_pix_objects = objects['xmin']
        x_max_pix_objects = objects['xmax']
        y_min_pix_objects = objects['ymin']
        y_max_pix_objects = objects['ymax']

        if i == n_start:
            print("\n (*) Plotting first reduced IMAGE frame ...")

            # plot 'first' reduced image
            fig3, ax = plt.subplots(figsize=(6, 6), num=3)
            plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
            m, s = np.mean(red_image), np.std(red_image)
            im = ax.imshow(red_image, interpolation='nearest',
                           cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
            tick_plot(ax, red_image[0], red_image[1])  # call tick function
            ax.grid(which='major', color='y', linestyle='-', linewidth=0.5)

            # over plot objects from source extraction
            for j in range(len(objects)):
                ellipse_plot(objects['x'][j], objects['y'][j], size,
                             objects['a'][j], objects['b'][j],
                             objects['theta'][j], 'none', 'red', [1])

            if do_plot:
                plt.savefig(out_dir+"extracted_sources.eps", dpi=300)

            # plt.title("Reduced CCD image of "+str(star_name))
            plt.show(block=False)
            plt.pause(0.1)

            if (len(objects) >= 1) and (len(objects) <= 2):
                print("\n     ! WARNING: not many sources have been detected")
                print("                perhaps source_sigma is too high? \n")
            elif len(objects) == 0:
                print("\n     !!! ERROR: no sources detected. Try using a smaller source_sigma.")
                print("                Exiting program in 10 seconds")
                plt.pause(10)
                exit()

            # take user input (screen click) for *TARGET* star's location
            while True:
                try:
                    toolbar = plt.get_current_fig_manager().toolbar
                    if toolbar.mode == '':
                        print("\n     >>> Please click on the image to choose the target star's co-ordinates:")
                    click = fig3.ginput(1, timeout=0, show_clicks=False)
                    init_pix_x = map(lambda x: x[0], click)
                    init_pix_y = map(lambda x: x[1], click)

                    # find star nearest to user's input
                    index = np.r_[(init_pix_x < x_max_pix_objects+pix_limit)
                                  & (init_pix_x > x_min_pix_objects-pix_limit)
                                  & (init_pix_y < y_max_pix_objects+pix_limit)
                                  & (init_pix_y > y_min_pix_objects-pix_limit)]

                except ValueError:
                    print("\n     !!! ERROR: you need to click on the screen to choose a target star")
                    continue

                # break loop if target found successfully
                if index.any() == True:  # changed from if X == True:
                    print("     > Found target star located at:")
                    print("     > (X,Y) = [ "+str('%.2f' % objects['x'][index])+" , "+str('%.2f' % objects['y'][index])+" ]")
                    break
                else:
                    toolbar = plt.get_current_fig_manager().toolbar
                    if toolbar.mode == '':
                        print("\n     !!! ERROR: no target object found at those co-ordinates")
                        continue

            # overplot target star aperture(s)
            ellipse_plot(objects['x'][index], objects['y'][index], size,
                         objects['a'][index], objects['b'][index],
                         objects['theta'][index], 'none', 'green',
                         [1.0, 1.5, 2.0])

            plt.pause(0.1)

            # take user input (screen click) for *COMPARISON* star's location
            while True:
                try:
                    toolbar = plt.get_current_fig_manager().toolbar
                    if toolbar.mode == '':
                        print("\n     >>> Please click on the image to choose the comparison star's co-ordinates:")
                    click = fig3.ginput(1, timeout=0, show_clicks=False)
                    comp_pix_x = map(lambda x: x[0], click)
                    comp_pix_y = map(lambda x: x[1], click)

                    # find star nearest to user's input
                    index_comp = np.r_[(comp_pix_x < x_max_pix_objects+pix_limit)
                                       & (comp_pix_x > x_min_pix_objects-pix_limit)
                                       & (comp_pix_y < y_max_pix_objects+pix_limit)
                                       & (comp_pix_y > y_min_pix_objects-pix_limit)]

                except ValueError:
                    print("\n     !!! ERROR: you need to click on the screen to choose a comparison star")
                    continue

                # continue loop if user chooses the comparison as the target
                if (objects['x'][index_comp] == objects['x'][index] and objects['y'][index_comp] == objects['y'][index]):
                    print("\n     ! WARNING: you have chosen the same source as target and comparison stars \n")
                    print("\n     >>> Do you really want to use the target as the comparison?")
                    CeqT = str((raw_input("     (y/n) = ")))
                    while CeqT not in (Y_proceed+N_proceed+Q_proceed):
                        print("\n     !!! ERROR: you need to type 'y' to proceed or 'n' to exit \n")
                        print("\n     >>> Do you really want to use the target as the comparison?")
                        CeqT = str((raw_input("     (y/n) = ")))
                    if CeqT in (N_proceed+Q_proceed):
                        print("\n     > You have chosen not to use the target star as the comparison star \n")
                        exit()
                    if CeqT in (Y_proceed):
                        print("\n     ! WARNING: differential magnitude will be zero in light curve since target and comparison stars are the same \n")
                        break

                # break loop if target found successfully
                if index_comp.any() == True:  # changed from if X == True:
                    print("     > Found comparison star located at:")
                    print("     > (X,Y) = [ "+str('%.2f' % objects['x'][index_comp])+" , "+str('%.2f' % objects['y'][index_comp])+" ]")
                    break
                else:
                    toolbar = plt.get_current_fig_manager().toolbar
                    if toolbar.mode == '':
                        print("\n     !!! ERROR: no comparison object found at those co-ordinates")
                        continue

            # overplot comparison star aperture
            ellipse_plot(objects['x'][index_comp], objects['y'][index_comp],
                         size, objects['a'][index_comp],
                         objects['b'][index_comp],
                         objects['theta'][index_comp], 'none', 'blue',
                         [1.0, 1.5, 2.0])

            # ask user for either a sky background or annulus subtraction
            sky_background = ""
            print("\n     >>> Do you want to calculate the sky background using annuli or a 'sky star'?")
            sky_background = str((raw_input("     (annuli/star) = ")))

            while sky_background not in ("annuli", "star", "quit"):
                print("\n !!! ERROR: you need to type 'annuli' or 'star' to proceed or 'quit' to exit \n")
                print("     >>> Do you want to calculate the sky background using annuli or a 'sky star'?")
                sky_background = str((raw_input("     (annuli/star) = ")))
            if sky_background in (Q_proceed):
                print("\n     > You have asked to quit: exiting program \n")
                exit()

            if sky_background == "star":
                # user defines input for *SKY-STAR BACKGROUND* location
                while True:
                    try:
                        toolbar = plt.get_current_fig_manager().toolbar
                        if toolbar.mode == '':
                            print("\n     >>> Please click on the image to choose the sky-star background co-ordinates:")
                        click = fig3.ginput(1, timeout=0, show_clicks=False)
                        sky_pix_x = map(lambda x: x[0], click)
                        sky_pix_y = map(lambda x: x[1], click)

                    except ValueError:
                        print("\n     !!! ERROR: you need to click on the screen to choose the sky-star background location")
                        continue

                    # check if sky location is near any sources
                    index_sky = np.r_[(sky_pix_x < x_max_pix_objects+pix_limit)
                                      & (sky_pix_x > x_min_pix_objects-pix_limit)
                                      & (sky_pix_y < y_max_pix_objects+pix_limit)
                                      & (sky_pix_y > y_min_pix_objects-pix_limit)]

                    # break loop if sky is not near to an extracted object
                    if index_sky.any() == True:  # cannot be if X == True:
                        print("\n     !!! ERROR: sky background aperture is too close to an object \n")
                        continue
                    else:
                        sky_pix_x, sky_pix_y = sky_pix_x[0], sky_pix_y[0]
                        break

                # overplot sky background aperture
                ellipse_plot(sky_pix_x, sky_pix_y, size, objects['a'][index],
                             objects['b'][index], objects['theta'][index],
                             'none', 'gold', [1.0])

                # check with user if OK to proceed:
                # If yes, override user position inputs with extracted centres
                print("     > Sky-Star background located at:")
                print("     > (X,Y) = [ "+str('%.2f' % sky_pix_x)+" , "+str('%.2f' % sky_pix_y)+" ] \n")

                print("     >>> Are you happy with your choice of target (green) and comparison (blue) stars and sky background (yellow)?")
                happy = str((raw_input("     (y/n) = ")))

                while happy not in (Y_proceed+N_proceed+Q_proceed):
                    print("\n     !!! ERROR: you need to type y to proceed or n to exit \n")
                    print("     >>> Are you happy with your choice of target (green) and comparison (blue) stars and sky background (yellow)?")
                    happy = str((raw_input("     (y/n) = ")))
                if happy in (N_proceed+Q_proceed):
                    print("\n     > You have specified that incorrect objects have been chosen: exiting program \n")
                    exit()

            plt.close(3)

            # define initial input locations for loop over images
            init_pix_x = objects['x'][index]
            init_pix_y = objects['y'][index]
            comp_pix_x = objects['x'][index_comp]
            comp_pix_y = objects['y'][index_comp]

        # for all remaining frames (after the 'first') find star location based on previous frame's locations
        else:
            # find closest source to target in previous frame
            index = np.r_[(init_pix_x < x_max_pix_objects)
                          & (init_pix_x > x_min_pix_objects)
                          & (init_pix_y < y_max_pix_objects)
                          & (init_pix_y > y_min_pix_objects)]

            # find closest source to comparison in previous frame
            index_comp = np.r_[(comp_pix_x < x_max_pix_objects)
                               & (comp_pix_x > x_min_pix_objects)
                               & (comp_pix_y < y_max_pix_objects)
                               & (comp_pix_y > y_min_pix_objects)]

        # skip frame if no stars are detected
        if len(objects[index]) < 1:
            print("     ! WARNING: no target star detected in frame "+str(i))
            do_targ = False

        if len(objects[index_comp]) < 1:
            print("     ! WARNING: no comparison star detected in frame "+str(i))
            do_comp = False

        # skip frame if multiple apertures are fitted
        if len(objects[index]) > 1:
            print("     ! WARNING: multiple apertures on target star in frame "+str(i))
            do_targ = False

        if len(objects[index_comp]) > 1:
            print("     ! WARNING: multiple apertures on comparison star in frame "+str(i))
            do_comp = False

        if do_targ:
            # warn user if CCD exceeds linearity (not really needed)
            if (objects['peak'][index] > 58000) and (objects['peak'][index] < 62000):
                print("     ! WARNING: target star flux is high: "+str('%i' % objects['peak'][index])+" counts in frame "+str(i)+"\n")
            elif objects['peak'][index] >= 62000:
                print("     ! WARNING: target star has saturated: "+str('%i' % objects['peak'][index])+" counts in frame "+str(i)+"\n")
                do_targ = False

        if do_comp:
            # skip frame if CCD saturates
            if (objects['peak'][index_comp] > 58000) and (objects['peak'][index_comp] < 62000):
                print("     ! WARNING: comparison star flux is high: "+str('%i' % objects['peak'][index_comp])+" counts in frame "+str(i)+"\n")
            elif objects['peak'][index_comp] >= 62000:
                print("     ! WARNING: comparison star has saturated: "+str('%i' % objects['peak'][index_comp])+" counts in frame "+str(i)+"\n")
                do_comp = False

        # if both target and comparison stars get errors, proceed to next frame
        if do_targ is False and do_comp is False:
            print("     ! WARNING: neither the target nor the comparison star were extracted in frame: "+str(i)+" (skipping frame) \n")
            continue

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # BELOW USES ADAPTIVE ELLIPTICAL APERTURES TO PRODUCE CURVE OF GROWTHS
        if i == n_start:

            # target curve of growth...
            for k in ap_size:

                flux, fluxerr, flag = sep.sum_ellipse(red_image,
                                                      objects['x'][index],
                                                      objects['y'][index],
                                                      int(k)*objects['a'][index],
                                                      int(k)*objects['b'][index],
                                                      objects['theta'][index], r=1.0,
                                                      err=bkg.globalrms,
                                                      gain=gain, mask=None)

                cg_flux.extend(flux)
                cg_area.extend(np.pi*int(k)*objects['a'][index]*int(k)*objects['b'][index])

                flux_comp, fluxerr_comp, flag_comp = sep.sum_ellipse(red_image,
                                                                     objects['x'][index_comp],
                                                                     objects['y'][index_comp],
                                                                     int(k)*objects['a'][index_comp],
                                                                     int(k)*objects['b'][index_comp],
                                                                     objects['theta'][index_comp],
                                                                     r=1.0, err=bkg.globalrms,
                                                                     gain=gain, mask=None)

                cg_flux_comp.extend(flux_comp)
                cg_area_comp.extend(np.pi*int(k)*objects['a'][index_comp]*int(k)*objects['b'][index_comp])

            # ask user for input on optimum aperture from curve of growth...
            fig4 = plt.figure(figsize=(6, 6), num=4)
            plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.15)
            grid = plt.GridSpec(10, 10, hspace=0.0, wspace=0.0)

            ax1 = fig4.add_subplot(grid[0:5, 0:], xticklabels=[])
            ax1.plot(cg_area, cg_flux, 'g-', zorder=1)
            ax1.scatter(cg_area, cg_flux, marker='o', facecolor='g',
                        edgecolor='k', s=200, label='target', zorder=2)
            ax1.set_xlim(-40, np.max(cg_area)+40)
            ax1.set_ylim(0, 1.1*np.max(cg_flux))
            ax1.legend(loc='lower right')
            ax1.set_ylabel("Total flux (Counts)")
            ax1.minorticks_on()
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')

            ax2 = fig4.add_subplot(grid[5:, 0:])
            ax2.plot(cg_area_comp, cg_flux_comp, 'b-', zorder=1)
            ax2.scatter(cg_area_comp, cg_flux_comp, marker='o', facecolor='b',
                        edgecolor='k', s=200, label='comparison', zorder=2)
            ax2.set_xlim(-40, np.max(cg_area)+40)
            ax2.set_ylim(0, 1.1*np.max(cg_flux_comp))
            ax2.legend(loc='lower right')
            ax2.set_xlabel("Aperture area (pixel$^2$)")
            ax2.set_ylabel("Total Flux (Counts)")
            ax2.minorticks_on()
            ax2.xaxis.set_ticks_position('both')
            ax2.yaxis.set_ticks_position('both')

            # plot aperture size labels
            for n1, n2, n3 in zip(cg_area[1:], cg_flux[1:], ap_size[1:]):
                ax1.text(n1, n2, str(n3), fontsize=10, color='w', va='center',
                         ha='center', zorder=3)

            for m1, m2, m3 in zip(cg_area_comp[1:], cg_flux_comp[1:], ap_size[1:]):
                ax2.text(m1, m2, str(m3), fontsize=10, color='w', va='center',
                         ha='center', zorder=3)

            # plt.title("Curve of Growth for varying aperture sizes")
            plt.show(block=False)
            plt.pause(0.1)

            if do_plot:
                plt.savefig(out_dir+"curve_of_growth.eps", dpi=300)

            # user chooses 'guess' for target ellipse from curve of growth
            while True:
                try:
                    print("\n     >>> Please choose the best aperture size for the (green) target star")
                    print("         (i.e. where the curve of growth becomes linear)")
                    cg_input = int(raw_input("         (int) = "))
                except ValueError:
                    print("\n     !!! ERROR: you need to input an INTEGER value \n")
                    continue

                if (cg_input < np.min(ap_size)) or (cg_input > np.max(ap_size)):
                    print("\n     !!! ERROR: you need to choose an integer aperture from those plotted \n")
                    continue
                else:
                    break

            # user chooses 'guess' for comparison ellipse from curve of growth
            while True:
                try:
                    print("\n     >>> Please choose the best aperture size for the (blue) comparison star")
                    print("         (i.e. where the curve of growth becomes linear)")
                    cg_input_comp = int(raw_input("         (int) = "))
                except ValueError:
                    print("\n     !!! ERROR: you need to input an INTEGER value \n")
                    continue

                if (cg_input_comp < np.min(ap_size)) or (cg_input_comp > np.max(ap_size)):
                    print("\n     !!! ERROR: you need to choose an integer aperture from those plotted \n")
                    continue
                else:
                    break

            plt.close(4)
            plt.pause(0.1)

            # aperture chosen based on index of ap_size array
            cg_input, cg_input_comp = int(cg_input), int(cg_input_comp)
            opt_ap_size = ap_size[cg_input]
            opt_ap_size_comp = ap_size[cg_input_comp]

            print("\n     > optimum aperture parameters for target:")
            print("     > semi-major and minor axes = [ "+str('%.2f' % float(opt_ap_size*objects['a'][index]))+" , "+str('%.2f' % float(opt_ap_size*objects['b'][index]))+" ] (pixels)")
            print("\n     > optimum aperture parameters for comparison:")
            print("     > semi-major and minor axes = [ "+str('%.2f' % float(opt_ap_size_comp*objects['a'][index_comp]))+" , "+str('%.2f' % float(opt_ap_size_comp*objects['b'][index_comp]))+" ] (pixels)")

            # print to screen so user is aware to be patient
            print("\n (*) Processing reduced FITS files")
            print("        ... please wait ...\n")

            # masking image for TARGET (i.e. mask all other sources)
            image_mask_targ = np.zeros(red_image.shape, dtype=np.bool)
            sep.mask_ellipse(image_mask_targ, objects['x'][~index],
                             objects['y'][~index],
                             opt_ap_size*objects['a'][~index],
                             opt_ap_size*objects['b'][~index],
                             objects['theta'][~index], r=1.0)
            index_mask_targ = np.where((image_mask_targ is True))
            masked_red_image_targ = np.copy(red_image)
            masked_red_image_targ[index_mask_targ] = np.nan

            # masking image for COMPARISON (i.e. mask all other sources)
            image_mask_comp = np.zeros(red_image.shape, dtype=np.bool)
            sep.mask_ellipse(image_mask_comp, objects['x'][~index_comp],
                             objects['y'][~index_comp],
                             opt_ap_size*objects['a'][~index_comp],
                             opt_ap_size*objects['b'][~index_comp],
                             objects['theta'][~index_comp], r=1.0)
            index_mask_comp = np.where((image_mask_comp is True))
            masked_red_image_comp = np.copy(red_image)
            masked_red_image_comp[index_mask_comp] = np.nan

            # masking image for ALL SOURCES (i.e. mask all other sources)
            image_mask_all = np.zeros(red_image.shape, dtype=np.bool)
            sep.mask_ellipse(image_mask_all, objects['x'], objects['y'],
                             opt_ap_size*objects['a'],
                             opt_ap_size*objects['b'],
                             objects['theta'], r=1.0)
            index_mask_all = np.where((image_mask_all is True))
            masked_red_image_all = np.copy(red_image)
            masked_red_image_all[index_mask_all] = np.nan

            # re-plot labelled sources using optimum apertures
            if do_plot:
                # plot 'first' reduced image (again)
                fig5, ax = plt.subplots(figsize=(6, 6), num=5)
                plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
                m, s = np.mean(red_image), np.std(red_image)
                im = ax.imshow(red_image, interpolation='nearest', cmap='gray',
                               vmin=m-s, vmax=m+s, origin='lower')
                # im = ax.imshow(masked_red_image_targ, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower') # plot masked comparison image
                # im = ax.imshow(masked_red_image_comp, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower') # plot masked comparison image
                tick_plot(ax, red_image[0], red_image[1])  # call tick function
                ax.grid(which='major', color='y', linestyle='-', linewidth=0.5)

                if sky_background == "annuli":
                    # target ellipses
                    ellipse_plot(objects['x'][index], objects['y'][index],
                                 opt_ap_size, objects['a'][index],
                                 objects['b'][index], objects['theta'][index],
                                 'none', 'green', [1.0, 1.5, 2.0])

                    # comparison ellipses
                    ellipse_plot(objects['x'][index_comp],
                                 objects['y'][index_comp], opt_ap_size_comp,
                                 objects['a'][index_comp],
                                 objects['b'][index_comp],
                                 objects['theta'][index], 'none', 'blue',
                                 [1.0, 1.5, 2.0])

                if sky_background == "star":
                    # target ellipse
                    ellipse_plot(objects['x'][index], objects['y'][index],
                                 opt_ap_size, objects['a'][index],
                                 objects['b'][index], objects['theta'][index],
                                 'none', 'green', [1.0])

                    # comparison ellipse
                    ellipse_plot(objects['x'][index_comp],
                                 objects['y'][index_comp], opt_ap_size_comp,
                                 objects['a'][index_comp],
                                 objects['b'][index_comp],
                                 objects['theta'][index], 'none', 'blue',
                                 [1.0])

                    # sky-star ellipse
                    ellipse_plot(sky_pix_x, sky_pix_y, opt_ap_size,
                                 objects['a'][index], objects['b'][index],
                                 objects['theta'][index], 'none', 'gold',
                                 [1.0])

                plt.savefig(out_dir+"labelled_sources.eps", dpi=300)
                plt.close(5)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # BELOW DOES APERTURE PHOTOMETRY USING CHOSEN OPTIMUM APERTURE SIZES...

        # Option 1) using a "sky star"...
        if sky_background == "star":
            try:
                # target star aperture photometry...
                if do_targ:
                    flux, fluxerr, flag = sep.sum_ellipse(red_image,
                                                          objects['x'][index],
                                                          objects['y'][index],
                                                          opt_ap_size*objects['a'][index],
                                                          opt_ap_size*objects['b'][index],
                                                          objects['theta'][index], r=1.0,
                                                          err=bkg.globalrms, gain=gain,
                                                          mask=image_mask_targ)

                    flux_sky, fluxerr_sky, flag_sky = sep.sum_ellipse(red_image,
                                                                      sky_pix_x, sky_pix_y,
                                                                      opt_ap_size*objects['a'][index],
                                                                      opt_ap_size*objects['b'][index],
                                                                      objects['theta'][index], r=1.0,
                                                                      err=bkg.globalrms, gain=gain,
                                                                      mask=image_mask_all)

                    # scale target flux by gain and correct for background
                    flux, flux_sky = gain*flux, gain*flux_sky
                    flux = flux - flux_sky
                    fluxerr = np.sqrt((fluxerr**2.0) + (fluxerr_sky**2.0))

                else:
                    flux, fluxerr = np.nan, np.nan

                # comparison star aperture photometry...
                if do_comp:
                    flux_comp, fluxerr_comp, flag_comp = sep.sum_ellipse(red_image,
                                                                         objects['x'][index_comp],
                                                                         objects['y'][index_comp],
                                                                         opt_ap_size_comp*objects['a'][index_comp],
                                                                         opt_ap_size_comp*objects['b'][index_comp],
                                                                         objects['theta'][index_comp],
                                                                         r=1.0, err=bkg.globalrms,
                                                                         gain=gain, mask=image_mask_comp)

                    flux_comp_sky, fluxerr_comp_sky, flag_comp_sky = sep.sum_ellipse(red_image,
                                                                                     sky_pix_x, sky_pix_y,
                                                                                     opt_ap_size_comp*objects['a'][index_comp],
                                                                                     opt_ap_size_comp*objects['b'][index_comp],
                                                                                     objects['theta'][index_comp],
                                                                                     r=1.0, err=bkg.globalrms,
                                                                                     gain=gain, mask=image_mask_all)

                    # scale comparison flux by gain and correct for background
                    flux_comp, flux_comp_sky = gain*flux_comp, gain*flux_comp_sky
                    flux_comp = flux_comp - flux_comp_sky
                    fluxerr_comp = np.sqrt((fluxerr_comp**2.0) + (fluxerr_comp_sky**2.0))

                else:
                    flux_comp, fluxerr_comp = np.nan, np.nan

            except ZeroDivisionError:
                print("     !!! ERROR: ZeroDivisionError in sep.sum_ellipse() when using sky-star")
                print("                Your chosen aperture sizes may be too large")
                continue


        ### Option 2) using "bkgann" as an annulus subtraction...
        if sky_background == "annuli":
            try:
                if do_targ:
                    flux, fluxerr, flag = sep.sum_ellipse(red_image,
                                                          objects['x'][index],
                                                          objects['y'][index],
                                                          opt_ap_size*objects['a'][index],
                                                          opt_ap_size*objects['b'][index],
                                                          objects['theta'][index], r=1.0,
                                                          err=bkg.globalrms, gain=gain,
                                                          mask=image_mask_targ,
                                                          bkgann=(1.5*opt_ap_size*objects['a'][index],
                                                                  2.0*opt_ap_size*objects['a'][index]))
                    flux, fluxerr = gain*flux, gain*fluxerr
                else:
                    flux, fluxerr = np.nan, np.nan

                if do_comp:
                    flux_comp, fluxerr_comp, flag_comp = sep.sum_ellipse(red_image,
                                                                         objects['x'][index_comp],
                                                                         objects['y'][index_comp],
                                                                         opt_ap_size*objects['a'][index_comp],
                                                                         opt_ap_size*objects['b'][index_comp],
                                                                         objects['theta'][index_comp],
                                                                         r=1.0, err=bkg.globalrms,
                                                                         gain=gain, mask=image_mask_comp,
                                                                         bkgann=(1.5*opt_ap_size*objects['a'][index_comp],
                                                                                 2.0*opt_ap_size*objects['a'][index_comp]))
                    flux_comp, fluxerr_comp = gain*flux_comp, gain*fluxerr_comp
                else:
                    flux_comp, fluxerr_comp = np.nan, np.nan

            except ZeroDivisionError:
                print(" !!! ERROR: ZeroDivisionError in sep.sum_ellipse() when using annuli")
                print(" Your chosen aperture sizes (and annuli) are probably too large")
                continue

        # convert fluxes to magnitudes and calculate errors
        mag = -2.5*np.log10(flux)
        mag_err = (2.5*0.43429448*fluxerr) / flux

        mag_comp = -2.5*np.log10(flux_comp)
        mag_comp_err = (2.5*0.43429448*fluxerr_comp) / flux_comp

        # append floats to time series
        final_time_HJD.append(midpoint_HJD)
        final_time_BJD.append(midpoint_BJD)
        targ_mag.append(float(mag))
        targ_mag_err.append(float(mag_err))
        comp_mag.append(float(mag_comp))
        comp_mag_err.append(float(mag_comp_err))
        targ_airmass.append(float(X))

        # re-define (X,Y) positions for sources in case of telescope jitter
        change_pix_x = (init_pix_x - 0.5*(x_min_pix_objects[index] + x_max_pix_objects[index]))
        change_pix_y = (init_pix_y - 0.5*(y_min_pix_objects[index] + y_max_pix_objects[index]))
        if sky_background == "star" and do_targ is True:
            sky_pix_x = sky_pix_x + change_pix_x
            sky_pix_y = sky_pix_y + change_pix_y
        if do_targ:
            init_pix_x = 0.5*(x_min_pix_objects[index] + x_max_pix_objects[index])
            init_pix_y = 0.5*(y_min_pix_objects[index] + y_max_pix_objects[index])
        if do_comp:
            comp_pix_x = 0.5*(x_min_pix_objects[index_comp] + x_max_pix_objects[index_comp])
            comp_pix_y = 0.5*(y_min_pix_objects[index_comp] + y_max_pix_objects[index_comp])

    plt.close('all')

    print("        ... DONE!")
    print("\n     > Length of observing block is "+str('%.4f' % (i*(exp_time+read_time)*24.0))+" (hrs) \n")


###############################################################################
###############################################################################
# BELOW DOES THE AIRMASS CORRECTION, DIFFERENTIAL LIGHT CURVES, SIGMA-CLIPPING

# calculate airmass correction
if ext_coeff:
    print(" (*) Including extinction correction using user-defined coefficient of: "+str(ext_coeff)+"\n")
    targ_mag_corr = [a-(b*ext_coeff) for a, b in zip(targ_mag, targ_airmass)]
    comp_mag_corr = [a-(b*ext_coeff) for a, b in zip(comp_mag, targ_airmass)]

# calculate differential magnitude (and errors) using list comprehension
diff_mag = [a-b for a, b in zip(targ_mag, comp_mag)]
diff_mag_err = [np.sqrt((a**2.0)+(b**2.0)) for a, b in zip(targ_mag_err, comp_mag_err)]

# sigma-clip outliers if do_clip is set to True
if do_clip:
    # convert to arrays
    final_time_BJD = np.array(final_time_BJD)
    diff_mag = np.array(diff_mag)
    diff_mag_err = np.array(diff_mag_err)
    targ_mag = np.array(targ_mag)
    targ_mag_err = np.array(targ_mag_err)
    comp_mag = np.array(comp_mag)
    comp_mag_err = np.array(comp_mag_err)
    final_time_HJD = np.array(final_time_HJD)
    if ext_coeff:
        targ_mag_corr = np.array(targ_mag_corr)
        comp_mag_corr = np.array(comp_mag_corr)
    targ_airmass = np.array(targ_airmass)

    # find and remove nans
    final_time_BJD = final_time_BJD[~np.isnan(diff_mag)]
    diff_mag_err = diff_mag_err[~np.isnan(diff_mag)]
    targ_mag = targ_mag[~np.isnan(diff_mag)]
    targ_mag_err = targ_mag_err[~np.isnan(diff_mag)]
    comp_mag = comp_mag[~np.isnan(diff_mag)]
    comp_mag_err = comp_mag_err[~np.isnan(diff_mag)]
    final_time_HJD = final_time_HJD[~np.isnan(diff_mag)]
    if ext_coeff:
        targ_mag_corr = targ_mag_corr[~np.isnan(diff_mag)]
        comp_mag_corr = comp_mag_corr[~np.isnan(diff_mag)]
    targ_airmass = targ_airmass[~np.isnan(diff_mag)]
    diff_mag = diff_mag[~np.isnan(diff_mag)]  # needs to be last!

    # calculate upper and lower bounds for clipping
    block_up = np.mean(diff_mag) + (do_clip*np.std(diff_mag))
    block_lo = np.mean(diff_mag) - (do_clip*np.std(diff_mag))
    index_clip = np.r_[(block_lo <= diff_mag) & (diff_mag <= block_up)]

    # create new clipped arrays
    final_time_BJD_clip = final_time_BJD[index_clip]
    diff_mag_clip = diff_mag[index_clip]
    diff_mag_err_clip = diff_mag_err[index_clip]
    targ_mag_clip = targ_mag[index_clip]
    targ_mag_err_clip = targ_mag_err[index_clip]
    comp_mag_clip = comp_mag[index_clip]
    comp_mag_err_clip = comp_mag_err[index_clip]
    final_time_HJD_clip = final_time_HJD[index_clip]
    if ext_coeff:
        targ_mag_corr_clip = targ_mag_corr[index_clip]
        comp_mag_corr_clip = comp_mag_corr[index_clip]
    targ_airmass_clip = targ_airmass[index_clip]


# output time series to file
if instrument == "STE":
    t0_start = str(t0_start)
    t0_start = t0_start[0:-7]
    image_filename = str(t0_start)+"_STE3"

datafile_id = open(out_dir+str(star_name)+"_"+str(image_filename)+"_"+str(observatory)+".dat", 'w+')
if ext_coeff:
    data = np.array([final_time_BJD, diff_mag, diff_mag_err, targ_mag,
                    targ_mag_err, comp_mag, comp_mag_err, final_time_HJD,
                    targ_mag_corr, comp_mag_corr, targ_airmass])
    datafile_id.write("#BJD-2450000.0_(TDB)   diff_mag   diff_mag_err   target_mag   target_mag_err   comp_mag   comp_mag_err   HJD-2450000.0_(UTC)   targ_mag_corr   comp_mag_corr   target_airmass\n")
    np.savetxt(datafile_id, data.T,
               fmt=['%.8f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f',
                    '%.8f', '%.8f', '%.8f', '%.4f'], delimiter='   ')
else:
    data = np.array([final_time_BJD, diff_mag, diff_mag_err, targ_mag,
                    targ_mag_err, comp_mag, comp_mag_err, final_time_HJD,
                    targ_airmass])
    datafile_id.write("#BJD-2450000.0_(TDB)   diff_mag   diff_mag_err   target_mag   target_mag_err   comp_mag   comp_mag_err   HJD-2450000.0_(UTC)   target_airmass\n")
    np.savetxt(datafile_id, data.T,
               fmt=['%.8f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f',
                    '%.8f', '%.4f'], delimiter='   ')
datafile_id.close()


# plot final light curve if do_plot is set to True
if do_plot:
    fig6 = plt.figure(figsize=(8, 6), num=6)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.gca().invert_yaxis()
    if do_clip:
        plt.scatter(final_time_BJD, diff_mag, c='r')
        plt.scatter(final_time_BJD_clip, diff_mag_clip, c='k')
    else:
        plt.scatter(final_time_BJD, diff_mag, c='k')
    plt.xlabel("BJD - 2450000.0 (TDB)")
    plt.ylabel(r"$\Delta$mag")
    plt.savefig(out_dir+str(star_name)+"_"+str(image_filename)+"_"+str(observatory)+".eps", dpi=300)
    plt.close(6)


###############################################################################
# # # # # # # # # # # # # # # # END OF PROGRAM # # # # # # # # # # # # # # # #
###############################################################################
