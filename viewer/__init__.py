import DicomReader, DicomViewer
def viewSequence(image_batch):
    data_importer = numpy2vtk(image_batch)
    DicomViewer.DicomViewer(data=data_importer, isSingleView=True).viewSlice()

def numpy2vtk(image_batch):
    dims = image_batch.shape
    image_batch = image_batch.reshape((dims[0], dims[1], dims[2]))
    data_importer = DicomReader.DicomReader.numpy2vtk(image_batch)
    return data_importer