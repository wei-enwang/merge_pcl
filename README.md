<h2>Introduction</h2>
<p>
    In this project, I adopted an <a href="https://users.cs.duke.edu/~tomasi/papers/phillips/phillips3DIM07.pdf">improved version</a> of the <i>Iterative Closest Point</i> algorithm to merge the pointclouds of an object obtained from different perspectives. 
</p>
<h2>Process of merging clouds</h2>
<p>
    To establish the entire pointcloud of objects, our program is given several scenes of objects randomly scattered on a table. 
    Assuming that we can perfectly isolate an items' cloud from the background, each scene produces a single pointcloud for each object. 
    The process starts by aligning the clouds in the first and second scene and checks if they can be merged by my icp algorithm. If the merge is successful, 
    the combined cloud is stored as the pointcloud representing that particular item; otherwise, both clouds are stored as potential representation
    of the item. The same process applies to future scenes.
    As we encounter more scenes, more unsuccessful merges will contribute to more representations of an item. We differentiate different
    items also by the icp algorithm. If the error of the icp is too large, the two clouds are considered different objects, which would create a "new item" in our 
    library of pointclouds.
</p>
<p>
    Here's the details on how I align and merge clouds:<br>
    <ul>
    <li>Points are sampled from the larger cloud to make its size the same as the smaller one. (The tranformation matrix computed by the SVD method performs badly if I directly find the nearest neighbors from one cloud to another without this process.) The points selected are random so I perform the whole process several(5) times and take the one that produces the lowest error.</li>
    <li>As icp’s performance is reliant on the clouds’ initial orientation, I also perform icp on clouds that are formed by rotating one of the two clouds 180 degrees about the x, y, and z axes, leading to 4 clouds for each new view of an item. Thus, I end up running 4*5=20 icps for each item for each new scene.</li>
    <li>The error measure of the adjusted icp, fractional root mean SD (FRMSD), seems to align with the success of the alignment. I merge the two clouds if the FRMSD value of the alignment is below 0.003. 
    If 0.005 > FRMSD > 0.003, the new cloud is stored as a new representation of the item, while the cloud is classified as a new object if FRMSD is greater than 0.005.</li>
    </ul>
</p>
<p>
    This program accepts input 3d models in the form of commmon pointcloud formats (.obj, readable by trimesh). Run `sample_mesh` to organize pointcloud files into desired formats.
</p>
<h2>TODO</h2>
<p>
    <ul>
    <li>Write script to create split in every model category(code done)</li>
    <li>Write script to transform obj files into numpy readable format(done)</li>
    </ul>
</p>
