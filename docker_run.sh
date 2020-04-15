docker run --rm --name cancer \
-v ${HOME}/Projects/cancer/mount_folder/input:/input \
-v ${HOME}/Projects/cancer/mount_folder/output:/output \
-v ${HOME}/Projects/cancer/mount_folder/data:/data \
cancer-python3.7 $1
