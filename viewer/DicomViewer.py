import vtk


'''
@author: qizhong.lin@philips.com

the class DicomViewer
reads the dicom data from directory or sequence as vtkImageImport (which can be transformed from numpy array)
view the sequence in coronal direction, saggital direction and axial direction
view the sequence in volumn rendering

    show slice by slice via keypress
    keypress "u" or "+" which shows next slice
    keypress "d" or "-" which shows pre slice
    
    volume rendering 
    keypress "j/t" toggle between joystick (position sensitive) and trackball (motion sensitive)
    left mouse  rotate
    middle mouse    zoom

'''
class DicomViewer():
    def __init__(self, isSingleView = False, data=None, dcm_dir="../data/dicom/S10"):
        self.isSingleView = isSingleView
        self.dcm_dir = dcm_dir

        self.focus_slice = 0    # status for UP/DOWN on which slice XY or YZ or XZ

        renWin = vtk.vtkRenderWindow()
        renWin.SetSize(1024, 1024)
        self.renWin = renWin
        self.iren = vtk.vtkRenderWindowInteractor()

        if data:
            self.reader = data
        else:
            self.reader = self.importDcm()

    def importDcm(self):
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(self.dcm_dir)
        reader.SetDataByteOrderToLittleEndian()
        reader.Update()
        print "dimension = {0}".format(reader.GetOutput().GetDimensions())
        print "the number of bits per pixel = {0}".format(reader.GetBitsAllocated())
        print "(rescale-slope, rescale-offset) = ({0}, {1})".format(reader.GetRescaleSlope(), reader.GetRescaleOffset())
        print "spacing = {0}".format(reader.GetPixelSpacing())
        print "(height, width) = ({0}, {1})".format(reader.GetHeight(), reader.GetWidth())
        return reader

    def __cvt2ubyte(self, reader):
        reader.Update()
        x, y, z = reader.GetOutput().GetDimensions()
        self.orientation_dict = {   #sliceNum, sliceIdx
            0: [lambda viewer: viewer.SetSliceOrientationToXY, z, z/2],
            1: [lambda viewer: viewer.SetSliceOrientationToYZ, x, x/2],
            2: [lambda viewer: viewer.SetSliceOrientationToXZ, y, y/2]
        }
        min, max = reader.GetOutput().GetScalarRange()
        data = vtk.vtkImageShiftScale()
        data.SetInputConnection(reader.GetOutputPort())
        data.SetShift(-1.0 * min)
        data.SetScale(255.0 / (max - min))
        data.SetOutputScalarTypeToUnsignedChar()
        return data

    def __renderVolumn(self, shift, viewer_range):
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleSwitch())
        ren = vtk.vtkRenderer()
        ren.SetViewport(viewer_range)
        self.renWin.AddRenderer(ren)
        self.iren.SetRenderWindow(self.renWin)

        # # Create transfer mapping scalar value to opacity
        # opacityTransferFunction = vtk.vtkPiecewiseFunction()
        # opacityTransferFunction.AddPoint(20, 0.0)
        # opacityTransferFunction.AddPoint(255, 0.2)
        #
        # # Create transfer mapping scalar value to color
        # colorTransferFunction = vtk.vtkColorTransferFunction()
        # colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        # colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
        # colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
        # colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
        # colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0.0)
        opacityTransferFunction.AddPoint(50, 0.05)
        opacityTransferFunction.AddPoint(100, 0.1)
        opacityTransferFunction.AddPoint(150, 0.2)

        # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
        # to be of the colors red green and blue.
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(50, 1.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(100, 0.0, 1.0, 0.0)
        colorTransferFunction.AddRGBPoint(150, 0.0, 0.0, 1.0)

        # The property describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()

        # The mapper / ray cast function know how to render the data
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        volumeMapper.SetInputConnection(shift.GetOutputPort())

        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        ren.AddVolume(volume)
        self.renWin.Render()

    def __loop(self):
        self.iren.AddObserver('KeyPressEvent', self.__keypress)
        self.iren.AddObserver('MouseMoveEvent', self.__mousemove)
        self.iren.Initialize()
        self.iren.Start()

    def __keypress(self, obj, event):
        key = obj.GetKeySym()
        print 'key = {0}'.format(key)
        if key in ['c']:
            self.focus_slice = (self.focus_slice + 1) % 3
        v = self.orientation_dict[self.focus_slice]
        if key in ['u', 'KP_Add']:
            v[2] = (v[2] + 1) % v[1]
        if key in ['d', 'KP_Subtract']:
            v[2] = (v[2] - 1) % v[1]

        self.__updateSlice()

    def __mousemove(self, obj, event):
        xypos = self.iren.GetEventPosition()
        x, y = xypos[0], xypos[1]
        print '(x, y) = ({0}, {1})'.format(x, y)

    def __updateSlice(self):
        for i, viewer in enumerate(self.viewers):
            viewer.SetSlice(self.orientation_dict[i][2])
            self.textActors[i].SetInput(self.__formatText(self.orientation_dict[i][2], self.orientation_dict[i][1]))
            viewer.Render()

    def __addText(self, text, x, y):
        textActor = vtk.vtkTextActor()
        textActor.SetDisplayPosition(x, y)
        textActor.SetInput(text)
        textActor.GetTextProperty().SetColor(0.5, 0.5, 0.5)
        textActor.GetTextProperty().SetFontSize(36)
        return textActor
    def __formatText(self, sliceIdx, sliceNum):
        return 'slice = {0}/{1}'.format(sliceIdx, sliceNum)

    def viewSlice(self):
        data = self.__cvt2ubyte(self.reader)

        viewer_ranges = [
            (0.0, 0.5, 0.5, 1.0),
            (0.5, 0.5, 1.0, 1.0),
            (0.0, 0.0, 0.5, 0.5)
        ]
        text_pos = [
            (20, 980),
            (550, 980),
            (20,  470),
            (550, 460)
        ]
        if self.isSingleView:
            viewer_ranges = [(0.0, 0.0, 1.0, 1.0)]

        self.viewers = []
        self.textActors = []
        for i, viewer_range in enumerate(viewer_ranges):
            viewer = vtk.vtkImageViewer2()
            viewer.SetInputConnection(data.GetOutputPort())
            viewer.SetupInteractor(self.iren)
            viewer.SetRenderWindow(self.renWin)
            viewer.SetColorLevel(127)
            viewer.SetColorWindow(255)
            viewer.GetImageActor().RotateY(180)  # flip top-bottom
            #viewer.GetRenderer().SetBackground(0.1, 0.1, 0.1)
            viewer.GetRenderer().SetViewport(viewer_range)
            text = self.__formatText(self.orientation_dict[i][2], self.orientation_dict[i][1])
            textActor = self.__addText(text, text_pos[i][0], text_pos[i][1])
            viewer.GetRenderer().AddActor2D(textActor)
            self.orientation_dict[i][0](viewer)()
            self.viewers.append(viewer)
            self.textActors.append(textActor)

        self.__updateSlice()

        if not self.isSingleView:
            self.__renderVolumn(data, (0.5, 0.0, 1.0, 0.5))

        self.__loop()

    def renderVolumn(self):
        data = self.__cvt2ubyte(self.reader)
        self.__renderVolumn(data, (0.0, 0, 1.0, 1.0))
        self.iren.Initialize()
        self.iren.Start()



def traverseImageData(imageData):
    width, height, slice = imageData.GetDimensions()
    for z in xrange(slice):
        for y in xrange(height):
            for x in xrange(width):
                val = imageData.GetScalarComponentAsFloat(x, y, z, 0)

import time
def calcTime():
    reader = DicomViewer().importDcm()
    ts = time.clock()
    traverseImageData(reader.GetOutput())
    print time.clock()-ts, "seconds process time"



if __name__ == '__main__':
    # calcTime()
    # exit()

    '''
    volume rendering 
    keypress "j/t" toggle between joystick (position sensitive) and trackball (motion sensitive)
    left mouse  rotate
    middle mouse    zoom
    '''
    #DicomViewer().renderVolumn()

    '''
    show slice by slice via keypress
    keypress "u" or "+" which shows next slice
    keypress "d" or "-" which shows pre slice
    '''
    DicomViewer().viewSlice()

