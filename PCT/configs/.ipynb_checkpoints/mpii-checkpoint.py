dataset_info = dict(
    dataset_name='mpii',
    
    paper_info = dict(
    author='Andriluka, Mykhaylo and Pishchulin, Leonid and '
           'Gehler, Peter and Schiele, Bernt',
    title='2D Human Pose Estimation: New Benchmark and State of the Art Analysis',
    container='IEEE Conference on Computer Vision and Pattern Recognition (CVPR)',
    year='2014',
    homepage='http://human-pose.mpi-inf.mpg.de/',
)
,
    keypoint_info={
        0: dict(name='r ankle', id=0, color=[255, 0, 0], type='lower', swap='l ankle'),
        1: dict(name='r knee', id=1, color=[255, 0, 0], type='lower', swap='l knee'),
        2: dict(name='r hip', id=2, color=[255, 0, 0], type='lower', swap='l hip'),
        3: dict(name='l hip', id=3, color=[0, 0, 255], type='lower', swap='r hip'),
        4: dict(name='l knee', id=4, color=[0, 0, 255], type='lower', swap='r knee'),
        5: dict(name='l ankle', id=5, color=[0, 0, 255], type='lower', swap='r ankle'),
        6: dict(name='pelvis', id=6, color=[255, 255, 0], type='lower', swap=''),
        7: dict(name='thorax', id=7, color=[255, 255, 0], type='upper', swap=''),
        8: dict(name='upper neck', id=8, color=[255, 255, 0], type='upper', swap=''),
        9: dict(name='head top', id=9, color=[255, 255, 0], type='upper', swap=''),
        10: dict(name='r wrist', id=10, color=[255, 0, 0], type='upper', swap='l wrist'),
        11: dict(name='r elbow', id=11, color=[255, 0, 0], type='upper', swap='l elbow'),
        12: dict(name='r shoulder', id=12, color=[255, 0, 0], type='upper', swap='l shoulder'),
        13: dict(name='l shoulder', id=13, color=[0, 0, 255], type='upper', swap='r shoulder'),
        14: dict(name='l elbow', id=14, color=[0, 0, 255], type='upper', swap='r elbow'),
        15: dict(name='l wrist', id=15, color=[0, 0, 255], type='upper', swap='r wrist')
    },
    skeleton_info={
        0: dict(link=('l ankle', 'l knee'), id=0, color=[0, 0, 255]),
        1: dict(link=('l knee', 'l hip'), id=1, color=[0, 0, 255]),
        2: dict(link=('r ankle', 'r knee'), id=2, color=[255, 0, 0]),
        3: dict(link=('r knee', 'r hip'), id=3, color=[255, 0, 0]),
        4: dict(link=('l hip', 'pelvis'), id=4, color=[0, 0, 255]),
        5: dict(link=('r hip', 'pelvis'), id=5, color=[255, 0, 0]),
        6: dict(link=('pelvis', 'thorax'), id=6, color=[255, 255, 0]),
        7: dict(link=('thorax', 'upper neck'), id=7, color=[255, 255, 0]),
        8: dict(link=('upper neck', 'head top'), id=8, color=[255, 255, 0]),
        9: dict(link=('r shoulder', 'upper neck'), id=9, color=[255, 0, 0]),
        10: dict(link=('r shoulder', 'r elbow'), id=10, color=[255, 0, 0]),
        11: dict(link=('r elbow', 'r wrist'), id=11, color=[255, 0, 0]),
        12: dict(link=('upper neck', 'l shoulder'), id=12, color=[0, 0, 255]),
        13: dict(link=('l shoulder', 'l elbow'), id=13, color=[0, 0, 255]),
        14: dict(link=('l elbow', 'l wrist'), id=14, color=[0, 0, 255])
    },
    joint_weights=[1.5, 1.2, 1., 1., 1.2, 1.5, 1., 1., 1., 1., 1.5, 1.2, 1., 1., 1.2, 1.5],
    
    sigmas=[0.089, 0.087, 0.107, 0.107, 0.087, 0.089, 0.095, 0.1, 0.096, 0.026, 0.062, 0.072, 0.079, 0.079, 0.072, 0.062]
    )
