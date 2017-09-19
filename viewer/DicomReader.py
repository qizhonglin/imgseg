#!/usr/bin/env python

import SimpleITK as sitk
import vtk
import numpy as np

from DicomViewer import DicomViewer


'''
@author: qizhong.lin@philips.com

the class DicomReader
reads the dicom data from directory or sequence as numpy array
show the sequence with class DicomViewer

example:
DicomReader(numpy_arr=sphere_pre).viewSlice()
or
DicomReader(dcm_dir="../data/dicom/S10").viewSlice()

DicomReader(dcm_dir="../data/dicom/S10").renderVolumn()
'''
class DicomReader():

    def __init__(self, numpy_arr=None, dcm_dir="../data/dicom/S10"):
        self.dcm_dir = dcm_dir

        if numpy_arr is not None:
            self.numpy_arr = numpy_arr
        else:
            self.numpy_arr = self.getNdarray(self.importDcm())

    def importDcm(self):
        image = None
        reader = sitk.ImageSeriesReader()
        series_found = reader.GetGDCMSeriesIDs(self.dcm_dir)
        if len(series_found):
            for serie in series_found:
                print("\nSeries:", serie)

                # Get the Dicom filename corresponding to the current series
                dicom_names = reader.GetGDCMSeriesFileNames(self.dcm_dir, serie)

                print("\nFiles in series: ", dicom_names)

                if len(dicom_names):
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()

        DicomReader.printImage(image)
        return image

    def viewSlice(self):
        dataImporter = self.numpy2vtk(self.numpy_arr)
        DicomViewer(data=dataImporter).viewSlice()

    def renderVolumn(self):
        dataImporter = self.numpy2vtk(self.numpy_arr)
        DicomViewer(dataImporter).renderVolumn()

    @staticmethod
    def getNdarray(image):
        return sitk.GetArrayFromImage(image)

    @staticmethod
    def printImage(image):
        print("\nImage size: ", image.GetSize())
        print("\nImage origin: ", image.GetOrigin())
        print("\nImage spacing: ", image.GetSpacing())

    @staticmethod
    def numpy2vtk(data_matrix):
        type_dict = {
            np.dtype(np.uint8): lambda dataImporter: dataImporter.SetDataScalarTypeToUnsignedChar,
            np.dtype(np.int16): lambda dataImporter: dataImporter.SetDataScalarTypeToShort,
            np.dtype(np.uint16): lambda dataImporter: dataImporter.SetDataScalarTypeToUnsignedShort,
            np.dtype(np.int32): lambda dataImporter: dataImporter.SetDataScalarTypeToInt,
            np.dtype(np.float): lambda dataImporter: dataImporter.SetDataScalarTypeToFloat,
            np.dtype(np.float32): lambda dataImporter: dataImporter.SetDataScalarTypeToFloat,
            np.dtype(np.double): lambda dataImporter: dataImporter.SetDataScalarTypeToDouble,
        }

        dataImporter = vtk.vtkImageImport()
        data_string = data_matrix.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        type_dict[data_matrix.dtype](dataImporter)()
        dataImporter.SetNumberOfScalarComponents(1)
        slice, height, width = len(data_matrix), len(data_matrix[0]), len(data_matrix[0][0])
        dataImporter.SetDataExtent(0, width - 1, 0, height - 1, 0, slice - 1)
        dataImporter.SetWholeExtent(0, width - 1, 0, height - 1, 0, slice - 1)

        return dataImporter


'''
the following functions show how to accelerate the traverse of volumn
via numba.jit
'''
from numba import jit
@jit
def get_range(imageData):
    slice, height, width = len(imageData), len(imageData[0]), len(imageData[0][0])
    min = 10000000
    max = -10000000
    for z in xrange(slice):
        for y in xrange(height):
            for x in xrange(width):
                val = imageData[z, y, x]
                if (val < min): min = val
                if (val > max): max = val
    return (min, max)

@jit
def scale_image(imageData, newRange = (0, 255)):
    slice, height, width = len(imageData), len(imageData[0]), len(imageData[0][0])
    seq = np.zeros((slice, height, width))
    (min, max) = get_range(imageData)
    for z in xrange(slice):
        for y in xrange(height):
            for x in xrange(width):
                val = imageData[z, y, x]
                newVal = (val - min) * (newRange[1] - newRange[0]) / (max - min) + newRange[0]
                seq[z, y, x] = newVal
    return seq

@jit
def traverseImageData( imageData):
    (min, max) = get_range(imageData)
    print (min, max)
    seq = scale_image(imageData)
    (min, max) = get_range(seq)
    print (min, max)

import time
def calcTime():
    imageData = DicomReader.getNdarray(DicomReader().importDcm())
    ts = time.clock()
    traverseImageData(imageData)
    print time.clock() - ts, "seconds process time"

# inputImage = sitk.ReadImage( "../data/Image0001.dcm" )
# inputImage = sitk.ReadImage( "../data/timg.jpeg" )
# print(inputImage)
if __name__ == '__main__':
    # calcTime()
    # exit()
    DicomReader().viewSlice()
    #DicomReader().renderVolumn()

