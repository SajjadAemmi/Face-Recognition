import PyInstaller.__main__

PyInstaller.__main__.run([
    'tk_torch_test.py',
    '--name=VideoAnonymizer',
    '--onedir',
    '--clean',
    '--paths="C:/Users/deep1/anaconda3/envs/MultiFace/Lib/site-packages;"',
    # '--paths="/src/mtcnn_pytorch;"',
    # '--hidden-import=theano',
    # '--add-data="C:/Users/deep1/anaconda3/envs/MultiFace/Lib/site-packages/mxnet;."',
    # '--windowed',
])
