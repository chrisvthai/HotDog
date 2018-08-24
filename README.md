# Hotdog, or not hotdog?
An image classifier based on the 'hotdog v/s not hotdog' classifier from the TV show 'Silicon Valley' that uses a convolutional neural network using Tensorflow. This has been trained on a small set of images, so this isn't a perfect classifier by any means.

Here is some sample output of the python script, which makes use of matplotlib to plot images and the machine's guess.
![alt text](https://raw.githubusercontent.com/chrisvthai/HotDog/master/sample_out_1.png)
![alt text](https://raw.githubusercontent.com/chrisvthai/HotDog/master/sample_out_2.png)

To run this script on some of your images, you can download everything in this folder and run the script with the *predict_image* argument as follows:

`python3 hotdog_own.py --predict_image=hotdog.jpg`

If you want to edit the network itself and try training your own version, make sure you run the script as follows:

`python3 hotdog_own.py --train`
