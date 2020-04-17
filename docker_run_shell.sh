docker run -it --rm --name cancer \
-v ${HOME}/Projects/cancer/mount_folder/input:/input \
-v ${HOME}/Projects/cancer/mount_folder/output:/output \
-v ${HOME}/Projects/cancer/mount_folder/data:/data \
-p 7688:7687 \
-p 7475:7474 \
cancer-python3.7
