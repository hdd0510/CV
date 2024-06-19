### Installation for CNN-based method
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
   * Note: It is based on ubuntu os, if you use window and want to make lib path for this, please view the setup_linux.py in CNN_based/lib/nms directory to change the way of setup path for the import.
   
3. Inference:
   
   In ubuntu, you need to fill the path in the infer.sh file (for image) anf infer_video.sh (for video). Then run in terminal:
   ```
   bash infer.sh
   ```
   For window, just copy the script and paste then run in terminal
   
   With the model checkpoint that we trained, the config is weight 32 which can be found in experiment with the corresponding name of the model (model with darkpose method also used the same config with the normal one)
   
   
