# Phantom pack

A program to find and analyze the Calimetrix phantom pack vials in PDFF/Water DICOM images.

Analysis includes mean, median, standard deviation, min and max values, calculated from a span of  saved as a dictionary, plus a plot of the images with the ROIs and values overlayed

### Install
`git clone https://github.com/Resoundant/phantompack.git`

Python 3.10 or 3.11 recommended.

`pip install -r requirements.txt`

### Run
`python phantom_pack.py <input_directory>`

results will be in <input_directory>/phatompack_results/

To batch process a folder than contains multiple patient-exams:

`python phantom_pack_batch.py <top_directory>`

will loop over each subfolder in <top_directory> as if running individually

To cellect up a bunch of folders of data into a spreadhsheet after running a batch, do 

`python collate_data.py <top_directory>`

to create <top_directory>/output.xlsx

### Theory of operation
PhantomPack finder is meant to operate on a single patient-exam, contained within the provided <input_directory>.  This directory will be scanned and load DICOM files.  PDFF and water images are identified by their ImageType and SeriesDescription tags.  PDFF images will be paired with a Water image by their acquisition sequence, AcquisitionTime, and SliceLocation tags.  This is meant to  descriminate related series from one another.

Once PDFF/water are paired, the PhantomPack vials are found by searching the water images for five cicles, roughly 19mm in diameter and roughly in a row.  This is accomplished using OpenCV's HoughCircle method.  [The diameter, vial radius, and row tolerance are all adjustable parameters.] If the pack is successfully found, it will be stored along with pdff/water images as  "circles", containing the centerpoint and estimated vial radius.

From the circle centers found, ROIs will be drawn in the PDFF images, with a diameter of 13mm.  The midpoint of the pack is calculated from the span of images with circles present, and the mean, median, stddev, min and max values are calculated from the slices spanning 20mm (+/-10mm) of the pack center.  [ROI size, span to calculate, and pack center slice are all configurable.]

Results are stored in the <input_directory>/phantompack_results, in date-and-time stamped files.  The follwing files are generated:
- A .json file containing the composite mean, median, stddev, min, and max values, along with some image parameters (TE, TR, flip angle, echo train length, Manufacturer, MRI SW version, pulse sequence name, series numbers and descriptions)
- An image of all PDFF/water pairs, with the vial circle overlaid on water, the ROI overlaid on PDFF, and the per-slice PDFF means positioned near the vials.
- An image of the PDFF/Water pairs that were in the center span, and used for the analsys,
- A plot of mean PDFF +/- stddev from ROIs per-slice.  A pair of dashed vertical lines shows which slices were included in the output.
- A plot of median PDFF per-slice.

Notes:
- Finding vials in PDFF images is often unsuccessful, not only for the 0% pdff vial.  This is why we find them in the water and copy to PDFF.  **Verify the ROI is in the PDFF vial.**
- Packs and results are sorted image left-to-right, so the values in the packs follow this order.  Which is to say, results are not presented as least to greatest pdff.
- Mean and Median plots use a default -10 value for all images with no packs, this is not calculated from the images.


UPDATES_ VIJAY SHARMA
FRAMING Phantom_packs @ UW Madison



