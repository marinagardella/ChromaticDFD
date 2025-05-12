# ChromaticDFD

### Usage

To run the method, simply provide the path to the input document image. The script will process the image and save several outputs:

- `characters_0.png`, `characters_1.png`, `characters_2.png`: Visualization of the extracted characters (connected components) in each channel.
- `stds_0.png`, `stds_1.png`, `stds_2.png`: Shows each detected character in each channel with a value representing its standard deviation.
- `outliers_0.png`, `outliers_1.png`, `outliers_2.png`: Highlights characters whose standard deviations fall outside the fluctuation interval defined by significance level `a`.
- `words.png`: Displays the words extracted using the EAST text detector.
- `nfa_0.png`, `nfa_1.png`, `nfa_2.png`: Final word-level detection made in each channel based on a Number of False Alarms (NFA) threshold `t`.

All output images are saved in the current working directory.


An example on how to run the method is given below:
```
python main.py input_document.png -a 0.1 -t 0.1
```
Where the arguments are:
- `-a`:	Significance level used for detecting outliers based on standard deviation.
- `-t`: NFA threshold used for final word-level detection.

### Online demo

You can try the method online in the following <a href="https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000543">IPOL demo</a>.
