#!/usr/bin/env python

import vtk

'''
@author: qizhong.lin@philips.com

the class MrViewer
view MR sequence with 4 phase in axial direction

    show slice by slice via keypress
    keypress "u" or "+" which shows next slice
    keypress "d" or "-" which shows pre slice
'''
class MrViewer():

    def __init__(self,
                 isFlat=True,
                 datas=None,
                 dcm_dirs = [
                     "../data/dicom/S10",       # pre
                     "../data/dicom/S10",       # arterial
                     "../data/dicom/S10",       # vein
                     "../data/dicom/S10"        # delay
                 ]):
        self.isFlat = isFlat
        self.dcm_dirs = dcm_dirs

        self.sliceIdx = 0
        self.sliceNum = 0

        renWin = vtk.vtkRenderWindow()
        renWin.SetSize(1024, 1024)
        if self.isFlat:
            renWin.SetSize(4*512, 512)
        self.renWin = renWin
        self.iren = vtk.vtkRenderWindowInteractor()


        if datas:
            self.readers = datas
        else:
            self.readers = [self.importDcm(dcm_dir) for dcm_dir in self.dcm_dirs]

    def importDcm(self, dcm_dir):
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(dcm_dir)
        reader.SetDataByteOrderToLittleEndian()
        reader.Update()
        return reader

    def __cvt2ubyte(self, reader):
        reader.Update()
        self.sliceNum = reader.GetOutput().GetDimensions()[2]
        self.sliceIdx = self.sliceNum / 2
        min, max = reader.GetOutput().GetScalarRange()
        data = vtk.vtkImageShiftScale()
        data.SetInputConnection(reader.GetOutputPort())
        data.SetShift(-1.0 * min)
        data.SetScale(255.0 / (max - min))
        data.SetOutputScalarTypeToUnsignedChar()
        return data

    def __loop(self):
        self.iren.AddObserver('KeyPressEvent', self.__Keypress)
        self.iren.Initialize()
        self.iren.Start()

    def __Keypress(self, obj, event):
        key = obj.GetKeySym()
        print 'key = {0}'.format(key)
        if key in ['u', 'KP_Add']:
            self.sliceIdx = (self.sliceIdx + 1) % self.sliceNum
        if key in ['d', 'KP_Subtract']:
            self.sliceIdx = (self.sliceIdx - 1) % self.sliceNum

        self.__updateSlice()

    def __addText(self, text, x, y):
        textActor = vtk.vtkTextActor()
        textActor.SetDisplayPosition(x, y)
        textActor.SetInput(text)
        textActor.GetTextProperty().SetColor(0.5, 0.5, 0.5)
        textActor.GetTextProperty().SetFontSize(36)
        return textActor
    def __formatText(self, sliceIdx, sliceNum):
        return 'slice = {0}/{1}'.format(sliceIdx, sliceNum)

    def __updateSlice(self):
        for viewer in self.viewers:
            viewer.SetSlice(self.sliceIdx)
            viewer.Render()
        self.textActor.SetInput(self.__formatText(self.sliceIdx, self.sliceNum))

    def viewSlice(self):
        data = [ self.__cvt2ubyte(reader) for reader in self.readers]


        viewer_ranges = [
            (0.0, 0.5, 0.5, 1.0),
            (0.5, 0.5, 1.0, 1.0),
            (0.0, 0.0, 0.5, 0.5),
            (0.5, 0.0, 1.0, 0.5)
        ]
        text_pos = (20, 980)
        if self.isFlat:
            viewer_ranges = [
                (0.0, 0.0, 0.25, 1.0),
                (0.25, 0.0, 0.5, 1.0),
                (0.50, 0.0, 0.75, 1.0),
                (0.75, 0.0, 1.0, 1.0)
            ]
            text_pos = [20, 470]

        self.viewers = []
        for i, viewer_range in enumerate(viewer_ranges):
            viewer = vtk.vtkImageViewer2()
            viewer.SetInputConnection(data[i].GetOutputPort())
            viewer.SetupInteractor(self.iren)
            viewer.SetRenderWindow(self.renWin)
            viewer.SetColorLevel(127)
            viewer.SetColorWindow(255)
            viewer.GetImageActor().RotateY(180)  # flip top-bottom
            #viewer.GetRenderer().SetBackground(0.1, 0.1, 0.1)
            viewer.GetRenderer().SetViewport(viewer_range)
            self.viewers.append(viewer)

        # add text in the first viewer
        text = self.__formatText(self.sliceIdx, self.sliceNum)
        self.textActor = self.__addText(text, text_pos[0], text_pos[1])
        self.viewers[0].GetRenderer().AddActor2D(self.textActor)
        self.__updateSlice()

        self.__loop()

if __name__ == '__main__':


    '''
    show slice by slice via keypress
    keypress "u" or "+" which shows next slice
    keypress "d" or "-" which shows pre slice
    '''
    MrViewer().viewSlice()