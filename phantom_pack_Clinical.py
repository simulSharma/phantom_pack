##This is Jeremy code @ V R Sharma @ UW Madison Version

import os
import sys
import pydicom
import numpy as np 
import cv2
import datetime 
import copy
import json
import matplotlib.pyplot as plt
import time
from image_labels import identifying_labels
from statistics import mode
from load_dicoms import load_dicoms
from circle_grouping import find_circle_groups
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# debug flags
DEBUG_VERBOSE = True
DEBUG_PLOTS = False
MATCH_TRACE = False
# circle index helpers
CX = 0
CY = 1
CR = 2

# Phantom pack and analysis parameters
OUTPUT_DIR = "phantompack_results"
VIAL_RADIUS_MM = 19/2      # radius of the phantom pack vials
VIAL_SEP_MM = 31           # 20px*1.56mm/px
VIAL_SEP_TOLERANCE_MM = 6  
ROI_RADIUS_MM = 13/2       # radius of the phantom pack ROI
RADIUS_TOLERANCE_MM = 4    # only find circles VIAL_RADIUS +/- RADIUS_TOLERANCE
VERTICAL_ALIGNMENT_TOLERANCE_MM = 7
ANALYSIS_SPAN_MM = 20      # analyze a range of images centered at the midpoint
ANALYSIS_CENTER_MM = None  # center span at a specific location, None to use midpoint
TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
DICOM_TAG_LIST = [
    "PatientName",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "EchoTrainLength",
    "FlipAngle",
    "VariableFlipAngleFlag",
    "PulseSequenceName",
    "0019109C",
    "InstitutionName",
    "StationName",
    "AcquisitionDate",
    "AcquisitionTime",
]


class FWSeries:
    def __init__(self, series_number:int):
        self.series_number = series_number
        self.series_number_pdff = series_number
        self.series_number_water = None
        self.series_description_pdff = None
        self.series_description_water = None
        self.image_pairs:list[ImagePair] = []
        self.pack_midpoint:float|None = None
        self.pack_first_slice:int|None = None
        self.pack_last_slice:int|None = None
        self.pack_first_slice_loc:float|None = None
        self.pack_last_slice_loc:float|None = None


    def find_pack_midpoint(self) -> float | None:
        self.pack_midpoint = find_midpoint([x.location_full for x in self.image_pairs if x.has_circles()])
        return self.pack_midpoint

    # def number_of_slices_in_span(fw_series:FWSeries, span_mm: float, center_loc: float | None ) -> int:
    #     if center_loc is None:
    #         return 0
    #     min_loc = center_loc - span_mm / 2
    #     max_loc = center_loc + span_mm / 2
    #     slices_in_span = [x.location_full for x in fw_series.image_pairs if min_loc <= x.location_full <= max_loc]
    #     return len(slices_in_span)

    def create_rois(self, roi_radius):
        ''' Draw ROIs in center of all circles, if present'''
        # pairs_to_analyze = [x for x in fw_series.image_pairs if x.has_circles()]
        # for img_pair in pairs_to_analyze:
        for img_pair in self.image_pairs:
            if not img_pair.has_circles():
                img_pair.rois = []
                continue
            roi_rad_px = roi_radius/img_pair.water.PixelSpacing[0]
            img_pair.rois = create_rois_from_circles(img_pair.circles, roi_rad_px)
        return 

    def find_pack_locations(self):
        all_locs = [x.location_full for x in self.image_pairs]
        pack_locs = [x.location_full for x in self.image_pairs if x.has_circles()]
        self.pack_first_slice_loc = min(pack_locs)
        self.pack_last_slice_loc = max(pack_locs)
        self.pack_first_slice = all_locs.index(self.pack_first_slice_loc)
        self.pack_last_slice = all_locs.index(self.pack_last_slice_loc)

    def sort_data_by_sliceloc(self):
        ''' Re-orders the fw_sereies.image_paris list by slice location, ascending) '''
        self.image_pairs = sorted(self.image_pairs, key=lambda x: x.location)

class ImagePair:
    def __init__(self, pdff:pydicom.Dataset, water:pydicom.Dataset):
        self.pdff = pdff
        self.water = water
        self.circles = []
        self.rois = []
        self.location:int = -999
        self.location_full:float = -999.0
        self.pixel_spacing = water.PixelSpacing[0]
        

    def set_pixel_spacing(self):
        self.pixel_spacing = self.pdff.PixelSpacing[0]
        if self.pdff.PixelSpacing[0] != self.water.PixelSpacing[0]:
            logger.warning(f"Pixel spacing mismatch! {self.pdff.SeriesDescription}")

    def has_circles(self):
        return self.circles !=[]
    
    def has_rois(self):
        return self.rois != []

def phantom_pack(
        directory_path:str, 
        vial_radius = VIAL_RADIUS_MM,
        radius_tolerance = RADIUS_TOLERANCE_MM,
        vert_align_tol = VERTICAL_ALIGNMENT_TOLERANCE_MM,
        roi_radius = ROI_RADIUS_MM,
        span_mm=ANALYSIS_SPAN_MM,
    ) -> dict:
    '''
    process all pdff data in directory_path
    return a list of dictionaries containing results for each pdff/water pair
    '''

    # prepare output directory
    output_dir = os.path.join(directory_path, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    #todo: make sure directory exist and is writeable

    # load dicoms in directory
    logger.info("Loading files...")
    all_dicoms = load_dicoms(directory_path, turbo_mode=True)
    
    # label each dicom dataset according to identifying_labels
    label_datasets(all_dicoms)

    # find pdff/water pairs; img_packs = [[pair1],[pair2],...]
    fw_series_paired = find_fw_pairs(all_dicoms) #todo: rename variables to clarify
    if len(fw_series_paired) == 0:
        logger.warning(f"No PDFF/Water data found in {directory_path}")
        return {}
    logger.info(f'========================================')
    logger.info(f'Found {len(fw_series_paired)} pdff/water series')
    logger.info('')

    # save summary of loaded data: each pdff/water series and description
    log_seriesdata_to_file(output_dir, fw_series_paired)

    # save summary of data loaded but unknown (no matching label)
    log_unknowns_to_file(output_dir, all_dicoms)

    for fw_serie in fw_series_paired:
        if len(fw_serie.image_pairs) == 0: # no data
            continue
        logger.info("")
        logger.info(f"Processing PDFF series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")
        logger.info(f"     with WATER series {fw_serie.series_number_water} {fw_serie.series_description_water}")

        find_packs_in_images(
            fw_serie,
            vial_radius=vial_radius,
            radius_tolerance=radius_tolerance,
            vert_align_tol=vert_align_tol
        )
        fw_serie.sort_data_by_sliceloc()
        fw_serie.create_rois(roi_radius=roi_radius) # put ROIs from all found circles

        # COMPUTE STATISTICS
        fw_serie.pack_midpoint = fw_serie.find_pack_midpoint() #set fw_series.pack_midpoint
        fw_serie.find_pack_locations() # set first and last locations and indeces of pack 
        if fw_serie.pack_midpoint is None:
            logger.warning(f"No pack midpoint found for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")
            continue
        results = compute_and_save_results(span_mm, output_dir, fw_serie)
    return results

def compute_and_save_results(span_mm, output_dir, fw_serie) -> dict:
    # this code is a bit rigid in expecting regularly structed data (same number of packs, same pixels in each ROI, etc))
    # to avoid it crashing the whole works, if something doesn't finish, it will except and move on, saving no data or
    # partial data
    try: 
        stats_min_loc = fw_serie.pack_midpoint-span_mm/2
        stats_max_loc = fw_serie.pack_midpoint+span_mm/2
        composite_results = composite_statistics(fw_serie, stats_min_loc, stats_max_loc)
    except:
        logger.warning(f"Error computing comsposite statistics for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")

    try:
        # collect info about dataset
        image_info = get_image_info(fw_serie)
        # save canvas of all pdff water pairs with circles
        array_filepath = os.path.join(output_dir, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_allimg.png")
        plot_results(fw_serie.image_pairs, dest_filepath=array_filepath, display_image=False)
        # save image of just the selected slices
        array_filepath = os.path.join(output_dir, f"{image_info['PatientName']}_{image_info['SeriesNumber_pdff']}_selected.png")
        image_pairs_in_span = img_pairs_in_span(fw_serie, stats_min_loc, stats_max_loc)
        plot_results(image_pairs_in_span, dest_filepath=array_filepath, display_image=False)
    except:
        logger.warning(f"Error saving plots for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")

    try:
        # save plots of slice values
        plot_slice_values(fw_serie, vert_lines=[stats_min_loc, stats_max_loc], directory_path=output_dir)
        results = composite_results | image_info

        if results:
            file_path = os.path.join(output_dir, f"{results['PatientName']}_{results['SeriesNumber_pdff']}.json")
            with open(file_path, 'w') as file:
                json.dump(results, file, indent=4)
            logger.info(f"  JSON data saved to {file_path}")
        if MATCH_TRACE:
            trace_data = []
            for imgs in fw_serie.image_pairs:
                trace_data.append({
                    "pdff_trace" : f"Series {imgs.pdff.SeriesNumber}, Instance {imgs.pdff.InstanceNumber}",
                    "water_trace" : f"Series {imgs.water.SeriesNumber}, Instance {imgs.water.InstanceNumber}",
                    "pdff_filename" : f"Series {imgs.pdff.filename}",
                    "water_filename" : f"Series {imgs.water.filename}",
                })
            file_path = os.path.join(directory_path, OUTPUT_DIR, f"{results['PatientName']}_{results['SeriesNumber_pdff']}_trace.json")
            with open(file_path, 'w') as file:
                json.dump(trace_data, file, indent=4)
        return results
    except:
        logger.warning(f"Error computing per-slice for series {fw_serie.series_number_pdff} {fw_serie.series_description_pdff}")
        return {}

def log_unknowns_to_file(output_dir, all_dicoms):
    unknowns = [x for x in all_dicoms if x.image_label == "unknown"]
    unknown_series_uids = list(set([x.SeriesInstanceUID for x in unknowns]))
    with open(os.path.join(output_dir, f"summary_data_unknown.txt"), 'w') as f:
        for series_uid in unknown_series_uids:
            unk_series = [x for x in unknowns if x.SeriesInstanceUID == series_uid]
            unk_series_nums = list(set([x.SeriesNumber for x in unk_series]))
            unk_series_descs = list(set([x.SeriesDescription for x in unk_series]))
            f.write(f"Unknown series containing {len(unk_series)} images\n")
            f.write(f"Series numbers: {unk_series_nums}\n") 
            f.write(f"Series descriptions: {unk_series_descs}\n")
            f.write(f"\n")

def log_seriesdata_to_file(output_dir, fw_series_paired):
    with open(os.path.join(output_dir, f"summary_data_found.txt"), 'w') as f:
        for fw_series in fw_series_paired:
            if len(fw_series.image_pairs) == 0:
                continue
            f.write(f"PDFF  series {fw_series.series_number_pdff},  {fw_series.series_description_pdff}\n")
            f.write(f"WATER series {fw_series.series_number_water}, {fw_series.series_description_water}\n")
            f.write(f"\n")


def label_datasets(datasets:list[pydicom.Dataset]):
    for ds in datasets:
        label_dataset(ds)

def label_dataset(ds:pydicom.Dataset):
    for id in identifying_labels:
        if id["search_for"] in ds.get(id["search_in"]):
            ds.image_label = id["image_label"]
            ds.label_match = id["label_match"]
            return
    # no label matched, set to unknown so the field exists
    ds.image_label = "unknown"
    ds.label_match = "unknown"
    return


def img_pairs_in_span(fw_series:FWSeries, min_loc:float, max_loc:float):
    images_in_span = [x for x in fw_series.image_pairs if min_loc <= x.location_full <= max_loc]
    images_in_span = sorted(images_in_span, key=lambda x: x.location_full)
    return images_in_span



def slice_stats(img:ImagePair) -> dict:
    # Compute the mean, median, and standard dev of a pdff/water pair.
    # img should be a single slice of the img_pack_data, with pdff, water and rois
    # Output: pdff_means:[], pdff_stddevs:[], pdff_medians:[] - lists of mean, stddev, median
    # for the circles
    stats = {}
    # default values are -10
    if img.has_rois() is False:
        stats["pdff_means"] = [-10]*5 # HACK - fixed for 5 ROIs
        stats["pdff_medians"] = [-10]*5
        stats["pdff_stddevs"] = [0]*5
        return stats
    # apply rois to PDFF
    pdff_means = []
    pdff_medians = []
    pdff_stddevs = []
    for r in img.rois:
        # make a circle mask that can be applied to pdff
        mask = np.zeros(img.pdff.pixel_array.shape, dtype=np.uint8)
        cv2.circle(mask, (r[CX], r[CY]), r[CR], color=1, thickness=-1) # solid circle (thickness = -1) filled with  1
        # calculate mean and median
        mean_pdff   = masked_mean(  img.pdff.pixel_array, mask)
        median_pdff = masked_median(img.pdff.pixel_array, mask)
        stddev_pdff = masked_stddev(img.pdff.pixel_array, mask)
        pdff_means.append(mean_pdff)
        pdff_medians.append(median_pdff)
        pdff_stddevs.append(stddev_pdff)
    stats["pdff_means"] = pdff_means
    stats["pdff_medians"] = pdff_medians
    stats["pdff_stddevs"] = pdff_stddevs
    return stats

def composite_statistics(fw_series:FWSeries, stats_min_loc, stats_max_loc) -> dict:
    '''
    Calculate the composite stats for slices in range (min_loc, max_loc)
    Output (dict): means:[], stddevs:[], medians:[], mins:[], maxs:[], samples:[]
    '''
    # sort all circles and rois by x-coord
    # this is necessary so that when cast into an np array, all of the rois across
    # slices are properly grouped
    for img_pair in fw_series.image_pairs:
        if img_pair.has_circles():
            img_pair.circles = sorted(img_pair.circles, key=lambda x: x[CX])
        if img_pair.has_rois():
            img_pair.rois = sorted(img_pair.rois, key=lambda x: x[CX])

    # quickly check that same number of ROIs in all images
    num_rois_in_images = []
    for img_pair in fw_series.image_pairs:
        if img_pair.has_rois():
            num_rois_in_images.append(len(img_pair.rois))
    if len(set(num_rois_in_images)) > 1:
        logger.warning("WARNING: not all images have same number of ROIs! This may cause issues")

    # check that all of the rois are aligned across slices, by making sure their centers are withing the rois radius
    # so that we don't mix values from different ROIs
    for i in range(len(fw_series.image_pairs)-1):
        if fw_series.image_pairs[i].has_rois() and fw_series.image_pairs[i+1].has_rois():
            for j in range(len(fw_series.image_pairs[i].rois)-1):
                if abs(float(fw_series.image_pairs[i].rois[j][CX]) - float(fw_series.image_pairs[i+1].rois[j][CX])) > fw_series.image_pairs[i].rois[j][CR]:
                    logger.warning("WARNING: ROIs not aligned across slices!")
                if abs(float(fw_series.image_pairs[i].rois[j][CY]) - float(fw_series.image_pairs[i+1].rois[j][CY])) > fw_series.image_pairs[i].rois[j][CR]:
                    logger.warning("WARNING: ROIs not aligned across slices!")
            

    # to calc mean, create lists made up of all pixels value in rois across all slices in range
    num_rois_mode = mode(num_rois_in_images)
    masked_values = [[] for _ in range(num_rois_mode)] #create empty list of lists
    for img_pair in fw_series.image_pairs:
        if (float(img_pair.pdff.SliceLocation) > stats_max_loc) or (float(img_pair.pdff.SliceLocation) < stats_min_loc): 
            continue
        if not img_pair.has_rois():
            continue
        for roi_index, roi in enumerate(img_pair.rois):
            # make a circle mask that can be applied to pdff
            vals = get_values_in_roi(img_pair.pdff, roi)
            masked_values[roi_index].extend(vals)

    # todo: rigid, doesn't handle situations where rois have different number of pixels
    # cast into np.array to take mean of each row, where a row contains the values for rois across slices
    np_arr = np.array(masked_values)
    # calculate mean across slices for a given roi
    results_dict = {}
    results_dict['means']    = np.mean(np_arr, axis=1).tolist()
    results_dict['stddevs']  = np.std(np_arr, axis=1).tolist()
    results_dict['medians']  = np.median(np_arr, axis=1).tolist()
    results_dict['mins']     = np.min(np_arr, axis=1).tolist()
    results_dict['maxs']     = np.max(np_arr, axis=1).tolist()
    results_dict['samples']  = [np_arr.shape[1]]*5
    results_dict = renormalize_stats(results_dict)
    return results_dict

def renormalize_stats(results_dict:dict) -> dict:
    # check mean value of the middle vial; it should always be 30% (20-40) if we find it is >101, renormalize by dividing by 100
    if results_dict["means"][2] < 101:
        results_dict["renormalized"] = False
        return results_dict
    results_dict["renormalized"] = True
    for key in results_dict.keys():
        if key == "renormalized": continue
        if key == "samples": continue
        results_dict[key] = [x/100 for x in results_dict[key]]
    return results_dict

def get_values_in_roi(pdff, r) -> list:
    mask = np.zeros(pdff.pixel_array.shape, dtype=np.uint8)
    cv2.circle(mask, (r[CX], r[CY]), r[CR], 1, -1) # solid circle (thickness = -1) filled with  1
    vals = apply_mask(pdff.pixel_array, mask)
    return vals

def create_negative_image(img):
    return 255 - np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

def make_clipped_image(water_img) -> np.ndarray:
    ''' Blacks off the top .65 of an image '''
    water_matlike = np.matrix(water_img)
    cutoff = int(np.floor(water_matlike.shape[0]*.65))
    water_matlike[:cutoff,:] = 0
    normalized_img = np.uint8(cv2.normalize(water_matlike, None, 0, 255, cv2.NORM_MINMAX))
    return normalized_img

def find_packs_in_images(
        fw_series:FWSeries, 
        vial_radius,
        radius_tolerance = RADIUS_TOLERANCE_MM,
        vert_align_tol = VERTICAL_ALIGNMENT_TOLERANCE_MM,
        vial_separation = VIAL_SEP_MM,
        vial_sep_tolerance = VIAL_SEP_TOLERANCE_MM,
        ):
    ''' 
        Finds circles in pdff/water image pairs ammends the fw_series.image_pairs to include those circles
    '''
    for img_pair in fw_series.image_pairs:
        water_ds = img_pair.water
        water_img = water_ds.pixel_array
        px_size = img_pair.pixel_spacing
        min_radius, max_radius, min_vail_sep = vial_sizes_in_px(vial_radius, radius_tolerance, px_size)
        water_circles = circles_img_bottom(water_img, min_radius, max_radius, min_vail_sep)
        if water_circles is None:
            img_pair.circles = []
            if DEBUG_PLOTS:
                plot_image(water_img, name=str(water_ds.SliceLocation), waitkey=0)
            continue
        if DEBUG_PLOTS:
            # water_img = remove_outliers_mad(water_img, threshold=3.5)
            plot_circles_ndarray(water_img, water_circles, name=str(water_ds.SliceLocation), waitkey=0)
        # pack_circles = find_phantom_pack(
        #     water_circles, 
        #     num_circles=5,
        #     row_tolerance_px=vert_align_tol/px_size,
        #     expected_radius=(vial_radius/px_size, np.ceil(radius_tolerance/px_size)),
        #     expected_sep_px=(vial_separation/px_size, np.ceil(vial_sep_tolerance/px_size))
        # )
        num_circles_in_pack = 5
        water_circles = reshape_and_sort_circles(water_circles, num_circles_in_pack) # drop trivial first dimension
        pack_circles = find_circle_groups(
            water_circles,
            radius = vial_radius/px_size,
            spacing = vial_separation/px_size,
            num_circles_in_group = num_circles_in_pack, 
            radius_tol = radius_tolerance/vial_radius,
            linear_tol = vert_align_tol/vial_separation, 
            spacing_tol = vial_sep_tolerance/vial_separation)
        if pack_circles == []:
            continue
        # if len(pack_circles) > 1:
        #     img_pair.circles = pack_circles[0]
        # else: 
        #     img_pair.circles = pack_circles
        # drop trivial first dimension
        img_pair.circles = pack_circles[0]
        
    count_circles = [pair for pair in fw_series.image_pairs if pair.has_circles()]
    logger.info(f"  {len(count_circles)} slices contain phantom pack")
    # if len(count_circles) == 0:
    #     with open("no_packs_found.txt", "w") as f:
    #         f.write(f"Series {water_ds.SeriesNumber}, {water_ds.SeriesDescription}")
    return

def reshape_and_sort_circles(circles_in:np.ndarray, num_circles=5) -> list: #np.ndarray:
    # convert from mdarray to list, return empty list if fewsert than num_circles 
    if (circles_in.shape[1] < num_circles):
        return []
    # Sort circles by y-coordinate to facilitate grouping
    circles = np.uint16(np.around(circles_in))
    circles = circles[0,:]
    circles = sorted(circles, key=lambda c: c[1])
    return circles #np.ndarray(circles)

def find_fw_pairs(all_dicoms:list[pydicom.Dataset]) -> list[FWSeries]:
    # Here were are matchmaking by enforcing the water image_label is same as pdff "label_match" tag
    # AcquisitionTime and SeriesNumber are used to separate series
    # SliceLocating is used to separate images
    series_found = []
    waters =[x for x in all_dicoms if x.image_label.startswith('water')] 
    pdffs = [x for x in all_dicoms if x.image_label.startswith('pdff')]
    # split up pdff's by series number
    series_numbers = list(set([x.SeriesNumber for x in pdffs]))
    for sn in series_numbers:
        # my_img_pack_data = []
        pdffs_in_series = [x for x in pdffs if x.SeriesNumber == sn]
        fw_series = FWSeries(sn)
        for ff in pdffs_in_series:
            fw_series.series_description_pdff = ff.SeriesDescription
            wat_match_label  = [x for x in waters if x.image_label == ff.label_match]
            wat_same_acqtime = [x for x in wat_match_label if acq_time_inrange(ff.AcquisitionTime, x.AcquisitionTime)]
            wat_same_loc     = [x for x in wat_same_acqtime if int(x.SliceLocation) == int(ff.SliceLocation)] # avoid float precision issues by casting to int
            if len(wat_same_loc) == 0:
                logger.warning(f"WARNING: PDFF no water match: {ff.SeriesDescription}, SerNum {ff.SeriesNumber}, AcqTime {ff.AcquisitionTime}, Loc {ff.SliceLocation}")
            if len(wat_same_loc) > 1:
                logger.warning(f"WARNING: PDFF no water match: {ff.SeriesDescription} {ff.SeriesNumber} {ff.AcquisitionTime} {ff.SliceLocation}")
                for w in wat_same_loc:
                    logger.warning(f"  {w.SeriesDescription} {w.SeriesNumber} {w.AcquisitionTime} {w.SliceLocation}")
            if len(wat_same_loc) >= 1:
                fw_series.series_number_water = wat_same_loc[0].SeriesNumber
                fw_series.series_description_water = wat_same_loc[0].SeriesDescription
                img_pair = ImagePair(ff,wat_same_loc[0])
                img_pair.location = int(ff.SliceLocation)
                img_pair.location_full = float(ff.SliceLocation)
                fw_series.image_pairs.append(img_pair)
        series_found.append(fw_series)
    return series_found

def vial_sizes_in_px(vial_radius, radius_tolerance, px_size):
    min_radius = int(vial_radius/px_size) - int(radius_tolerance/px_size)
    max_radius = int(vial_radius/px_size) + int(np.ceil(radius_tolerance/px_size))
    min_vail_sep = vial_radius/px_size
    return min_radius,max_radius,min_vail_sep

def circles_img_bottom(img, min_radius, max_radius, min_vail_sep):
    cropped_img = make_clipped_image(img)
    circles = find_circles(cropped_img, minDist=min_vail_sep, minRadius=min_radius, maxRadius=max_radius)
    return circles


def sort_circles_by_x_coord(circles:np.ndarray):
    x_locs = []
    for c in circles:
        x_locs.append(c[0])
    x_locs = sorted(x_locs)
    sorted_circles = []
    for loc in x_locs:
        for c in circles:
            if c[0] == loc:
                sorted_circles.append(c)
                break
    return sorted_circles


# def sort_data_by_sliceloc(fw_series:FWSeries):
#     ''' Re-orders the fw_sereies.image_paris list by slice location, ascending) '''
#     fw_series.image_pairs = sorted(fw_series.image_pairs, key=lambda x: x.location)

def find_midpoint(locations: list) -> float | None:
    """Finds the midpoint of a list of values."""
    if not locations:
        return None
    locations = sorted(list(set(locations)))  # remove duplicates
    midpoint = (locations[0] + locations[-1]) / 2
    return midpoint

def number_of_slices_in_span(fw_series:FWSeries, span_mm: float, center_loc: float | None ) -> int:
    if center_loc is None:
        return 0
    span_min_loc = center_loc - span_mm / 2
    span_max_loc = center_loc + span_mm / 2
    slices_in_span = [x.location_full for x in fw_series.image_pairs if span_min_loc <= x.location_full <= span_max_loc]
    return len(slices_in_span)


def create_rois_from_circles(circles, roi_radius_px) -> list:
    rois = copy.deepcopy(circles)
    roi_radius_px = np.uint8(roi_radius_px)
    for i in range(len(rois)):
        rois[i][2] = roi_radius_px
    return rois

def print_mean_median_values(pdff_means, pdff_medians):
    mean_str = "Mean values: "
    median_str = "Median values: "
    pdff_means = sorted(pdff_means)
    pdff_medians = sorted(pdff_medians)
    for mn in pdff_means:
        mean_str += f"{mn:.2f}, "
    for md in pdff_medians:
        median_str += f"{md:.2f}, "
    print(mean_str)
    print(median_str)

def find_imagetype(dicoms:list[pydicom.Dataset], contrast:str) -> list[pydicom.Dataset]:
    images = []
    for ds in dicoms:
        if contrast in ds.ImageType:
            images.append(ds)
    return images


def acq_time_inrange(ff_acqtime, wat_acqtime, rng=1):
    return ((int(wat_acqtime) >= int(ff_acqtime)-rng) and (int(wat_acqtime) <= int(ff_acqtime)+rng))



def find_circles(img:np.ndarray, minDist:float=0.01, param1:float=300, param2:float=10, minRadius:int=2, maxRadius:int=20):
    # # docstring of HoughCircles: 
    # # HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # HOUGH_GRADIENT_ALT is supposed to be more accurate but it doesn't find any circles
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist, param1=param1, param2=0.9, minRadius=minRadius, maxRadius=maxRadius)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=0.01, param1=300, param2=0.99)
    return circles

def find_phantom_pack(
        circles_in, 
        num_circles:int = 5, row_tolerance_px = 5, 
        expected_radius:tuple[float,float]|None = None, 
        expected_sep_px:tuple[float,float]|None = None
        ) -> list:
    """
    Finds the  group of 5 circles that lie roughly in a horizontal line.

    Parameters:
        circles (list of lists): A list where each element is [x_center, y_center, radius].
        min_num_circles (int): The minimum number of circles in a row keep.
        row_tolerance: The maximum difference in y-coordinates to consider as vertically aligned (in a row).
        expected_radius (tuple[float,float]): A tuple where the first element is the expected radius and the second element is the tolerance.
            discard circles that are not in expected_radius +/- tolerance
        expected_sep_px (tuple[float,float]): the (expected separation, tolerance) between centers of phantom 
    
    Returns:
        list: The group of circles in the horizontal line.
    """
    phantoms = []
    if (circles_in.shape[1] < num_circles):
        return []
    # Sort circles by y-coordinate to facilitate grouping
    circles = np.uint16(np.around(circles_in))
    circles = circles[0,:]
    circles = sorted(circles, key=lambda c: c[1])
    
    # Find all groups of circles that have similar y-coordinates
    # This allows that their may be more than one row of circles
    # bug: this will duplicate rows, leaving off first entry in each subsequent row
    horizontal_groups = []
    for i in range(len(circles)):
        group = [circles[i]]
        for j in range(i + 1, len(circles)):
            if abs(int(circles[j][1]) - int(circles[i][1])) <= row_tolerance_px:
                group.append(circles[j])
        horizontal_groups.append(group)

    # only keep horizontal groups with at least 5 circles
    phantoms = [g for g in horizontal_groups if len(g) >= num_circles]
    if phantoms == []:
        return []

    # only keep circles in expected_radius +/- px_tolerance
    if expected_radius != None:
        similar_radius = []
        for row in phantoms:
            g = []
            for c in row:
                radius = abs(float(c[2]) - expected_radius[0]) 
                if radius <= expected_radius[1]:
                    g.append(c)
            similar_radius.append(g)
        phantoms = similar_radius

    # keep only simarly-radius groups that meet minimum num circles
    phantoms = [g for g in phantoms if len(g) == num_circles]
    if phantoms == []:
        return []
    # HACK - At this point we're done messing aruond with this multiple row stuff; if there
    # is more than one row of 5 circles, we're only keeping the first 
    if len(phantoms) > 1:
        logger.warning(f"HACK - {len(phantoms)} found, keeping only first phantom in list")
    phantoms = phantoms[0]

    # only keep circles in expected_sep +/- px_tolerance
    if expected_sep_px != None:
        keep_indeces = []
        phantoms = sort_circles_by_x_coord(phantoms)
        for j in range(len(phantoms)-1):
            phantom_sep = float(phantoms[j+1][0]) - float(phantoms[j][0])
            if (abs(phantom_sep - expected_sep_px[0]) < expected_sep_px[1]):
                keep_indeces.append(j)
            # else:  
                # logger.info(f"found circle outlier")
        sorted(keep_indeces)
        phantom_tmp = []
        for i in keep_indeces:
            phantom_tmp.append(phantoms[i])
        phantoms = phantom_tmp


    if len(phantoms) != num_circles:
        # print(f"   ... so close - phantom has {len(phantoms)} circles, must be {num_circles}")
        return []
    return phantoms

def check_vial_spacing(ph, min_space, max_space) -> bool:
    for i in range(len(ph)-1):
        c0 = ph[i]
        c1 = ph[i+1]
        if ((abs(c0[CX]-c1[CX]) < min_space) or 
            (abs(c0[CX]-c1[CX]) > max_space)):
            return False
    return True

def avg_circle_spacing(phantoms):
    avg_spacing = 0
    for i in range(len(phantoms) - 1):
        avg_spacing += phantoms[i][2] - phantoms[i+1][2]
    avg_spacing = avg_spacing/(len(phantoms) - 1)
    return avg_spacing

def avg_circle_radius(phantoms):
    avg = 0
    for i in range(len(phantoms) - 1):
        avg += phantoms[i][0] - phantoms[i+1][0]
    avg = avg/(len(phantoms) - 1)
    return avg

def plot_image(img, name='image', waitkey=1):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def plot_circles_list(img, circles, name='image', waitkey=1):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles:
        cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)   
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def plot_circles_ndarray(img, circles, name='image', waitkey=1):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles[0,:]: # don't remembwer what packing needed this slice
        cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)   
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def plot_selected_image(img_data:dict, dest_filepath:str=None, display_image=False):
    ''' save pdff and water with ROIs on them'''
    # put rois onto pdff and water images
    cimg_water = np.uint8(cv2.normalize(img_data["water"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
    cimg_water = cv2.cvtColor(cimg_water, cv2.COLOR_GRAY2BGR)
    cimg_pdff = np.uint8(cv2.normalize(img_data["pdff"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
    cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(img_data["circles"]))
    for c in np_circles:
        cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(0,255,0),1)
        cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,255,0),1)
    np_rois = np.uint16(np.around(img_data["rois"]))
    for c in np_rois:
        cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(0,0,255),1)
        cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,0,255),1)  
    # canvas to have both pdff (left) and water (right) in one image
    height, width, channels = cimg_water.shape
    canvas = np.zeros(( width, 2 * height, channels), dtype=np.uint8)
    canvas[:height, :width] = cimg_pdff
    canvas[:height, width:] = cimg_water

    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Selected Image", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_array(img_pack_data:list[dict], dest_filepath:str=None,display_image=False):
    """Plot an array of images using OpenCV."""
    cols = 5
    rows = np.uint8(np.ceil(len(img_pack_data) / cols))
    # Create a blank canvas to hold the images
    # cimg_setup = np.uint8(img_pack_data[0]["water"].pixel_array)
    cimg_setup = cv2.cvtColor(np.uint8(img_pack_data[0]["water"].pixel_array), cv2.COLOR_GRAY2BGR) #bug: assumes all images same resolution
    height, width, channels = cimg_setup.shape

    canvas = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)
    for i, mydict in enumerate(img_pack_data):
        cimg = np.uint8(cv2.normalize(mydict["water"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
        cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
        if mydict["circles"] is not None:
            np_circles = np.uint16(np.around(mydict["circles"]))
            for c in np_circles:
                cv2.circle(cimg,(c[0],c[1]),c[2],(255,255,0),1)             # draw the outer circle

        # Place each image on the canvas
        row = i // cols
        col = i % cols
        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = cimg
        # Display the canvas
    
    # save image
    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def plot_results(image_pairs:list[ImagePair], dest_filepath:str="", display_image=False):
    """Plot PDFF and water images with ROIs using OpenCV."""
    cols = 2
    rows = np.uint32(len(image_pairs))
    # Create a blank canvas to hold the images
    cimg_setup = cv2.cvtColor(np.uint8(image_pairs[0].water.pixel_array), cv2.COLOR_GRAY2BGR) #bug: assumes all images same resolution
    height, width, channels = cimg_setup.shape

    canvas = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)
    for i, img_pair in enumerate(image_pairs):
        cimg_water = np.uint8(cv2.normalize(img_pair.water.pixel_array, None, 0, 255, cv2.NORM_MINMAX))
        cimg_water = cv2.cvtColor(cimg_water, cv2.COLOR_GRAY2BGR)
        cimg_pdff = np.uint8(cv2.normalize(img_pair.pdff.pixel_array, None, 0, 255, cv2.NORM_MINMAX))
        cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
        if img_pair.has_circles():
            np_circles = np.uint16(np.around(img_pair.circles))
            for c in np_circles:
                cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,0,255),1)             # draw the outer circle
        if img_pair.has_rois():
            np_rois = np.uint16(np.around(img_pair.rois))
            mystats = slice_stats(img_pair)
            for j, c in enumerate(np_rois):
                cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(255,255,0),1)
                mystr = f"{mystats['pdff_means'][j]:.1f}"
                text_size, _ = cv2.getTextSize(mystr, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                text_w, text_h = text_size
                mypt = (c[0]-3*c[2],c[1]+4*c[2]+text_h) # default/odd, plot below vial
                if (j % 2 == 0): #even, plot above  vial 
                    mypt = (c[0]-3*c[2],c[1]-4*c[2])
                
                cv2.rectangle(cimg_pdff, (mypt[0], mypt[1]), (mypt[0] + text_w, mypt[1] - text_h), (0,0,0), -1)
                cv2.putText(cimg_pdff, mystr, mypt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
        # plot slice location on bottom of pdff image
        loc_nodecimals = f"{float(img_pair.pdff.get('SliceLocation')):.1f}"
        loc_str = f"LOC: {loc_nodecimals}"
        loc_fontscale = 0.6
        loc_size, _ = cv2.getTextSize(loc_str, cv2.FONT_HERSHEY_SIMPLEX, loc_fontscale, 1)
        loc_w, loc_h = loc_size
        loc_pt = (int(width/2 - loc_w/2), 2*loc_h)
        cv2.rectangle(cimg_pdff, loc_pt, (loc_pt[0] + loc_w, loc_pt[1] - loc_h), (0,0,0), -1)
        cv2.putText(cimg_pdff, loc_str, loc_pt, cv2.FONT_HERSHEY_SIMPLEX, loc_fontscale, (255,255,0), 1)
        # Place each image on the canvas
        canvas[i * height:(i + 1) * height,     0:width  ] = cimg_water
        canvas[i * height:(i + 1) * height, width:width*2] = cimg_pdff
    
    # save image
    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def find_closest_value(mylist:list, target_value):
    """Finds the number in a list closest to a given target value."""
    return min(mylist, key=lambda x: abs(x - target_value))

def apply_mask(image, mask) -> list:
    np_img = np.array(image)
    vals = np_img[mask == 1].tolist() # need to be a list? or leave as np.array??
    return vals

def masked_mean(image, mask) -> float:
    vals = apply_mask(image, mask)
    if len(vals) == 0:
        return None
    return sum(vals) / len(vals)

def masked_median(image, mask) -> float:
    vals = apply_mask(image, mask)
    if len(vals)  == 0:
        return None
    if len(vals) % 2 == 0:
        return (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    return  vals[len(vals) // 2 ] 

def masked_stddev(image, mask) -> float:
    vals = apply_mask(image, mask)
    if len(vals)  == 0:
        return None
    return  np.std(vals)

def get_image_info(fw_series:FWSeries) -> dict:
    info = {}
    pdff  = fw_series.image_pairs[0].pdff
    water = fw_series.image_pairs[0].water
    for tag in DICOM_TAG_LIST:
        info[tag] = str(pdff.get(tag))
    info["SeriesDescription_pdff"] = pdff.get("SeriesDescription")
    info["SeriesDescription_water"] = water.get("SeriesDescription")
    info["SeriesNumber_pdff"] = pdff.get("SeriesNumber")
    info["SeriesNumber_water"] = water.get("SeriesNumber")
    return info

def plot_slice_values(fw_series:FWSeries, vert_lines=[], directory_path=''):
    # each entry in this will will be the 5 vials  in a slice
    pdff_means = []
    pdff_medians = []
    pdff_stddevs = []
    slice_locations = []

    for img in fw_series.image_pairs:
        mystats = slice_stats(img)
        pdff_means.append(mystats['pdff_means'])
        pdff_medians.append(mystats['pdff_medians'])
        pdff_stddevs.append(mystats['pdff_stddevs'])
        slice_locations.append(img.location_full)
    #convert to np.array and transpose, so that each row is one roi loc across slices
    data_means = np.array(pdff_means).transpose()
    data_medians = np.array(pdff_means).transpose()
    data_stddevs = np.array(pdff_stddevs).transpose()


    for i in range(data_means.shape[0]):
        plt.errorbar(slice_locations, data_means[i],  yerr=data_stddevs[i], fmt='.', label=f"Mean {i}")
    if len(vert_lines) > 0:
        plt.axvline(x=vert_lines[0],  color='r', linestyle='--', linewidth=1) # vertical lines at edge of selected slices
        plt.axvline(x=vert_lines[-1], color='r', linestyle='--', linewidth=1)
    plt.title(f"PDFF Means +/- StdDev {fw_series.series_number_pdff} - {fw_series.series_description_pdff}")
    plt.xlabel("Slice Location")
    plt.ylabel("Mean PDFF")
    plt.legend()
    plt.grid(True)
    # plt.show()
    filename = f"{fw_series.image_pairs[0].pdff.PatientName}_{fw_series.series_number_pdff}_mean_stddev.png"
    plt.savefig(os.path.join(directory_path, filename))
    plt.close()

    for i in range(data_means.shape[0]):
        plt.plot(slice_locations, data_medians[i], '-x', label=f"Median {i}")
    if len(vert_lines) > 0: 
        plt.axvline(x=vert_lines[0],  color='r', linestyle='--', linewidth=1) # vertical lines at edge of selected slices
        plt.axvline(x=vert_lines[-1], color='r', linestyle='--', linewidth=1)
    plt.title(f"Median PDFFs per slice {fw_series.series_number_pdff} - {fw_series.series_description_pdff}")
    plt.xlabel("Slice Location")
    plt.ylabel("Median PDFF")
    plt.legend()
    plt.grid(True)
    # plt.show()
    filename = f"{fw_series.image_pairs[0].pdff.PatientName}_{fw_series.series_number_pdff}_median.png"
    plt.savefig(os.path.join(directory_path, filename))
    plt.close()
    

def remove_outliers_mad(image, threshold=3.5):
    ''' introduced this function to try to deal with background white pixels.
        It actually did an admirable job of clipping those, but it also broke
        the ability to detect the vials.  Unused but left for educational purposes.
    '''
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    # MAD to standard deviation approximation (if needed): σ ≈ 1.4826 * MAD
    modified_z_scores = 0.6745 * (image - median) / (mad + 1e-6)
    mask = np.abs(modified_z_scores) < threshold
    # num_outliers = np.sum(mask==False)
    # if num_outliers > 0:
    #     logging.info(f"Removed {num_outliers} outliers from image")
    return np.where(mask, image, 0)  # replace outliers with 0 


if __name__ == "__main__":
    directory_path = sys.argv[1]
    logfile = os.path.join(directory_path, "phantom_pack.log")
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger.info(f"Processing {directory_path}")
    results = phantom_pack(directory_path)
    

    # testimg_path = r'C:\testdata\PhantomPack\PQ024\SER00090\IMG00019.dcm'
    # directory_path = os.path.dirname(testimg_path)
    # ds = pydicom.dcmread(testimg_path)
    # circles_in_pdff(ds, min_radius=1, max_radius=60, min_sep=1)


