So this project aims to be the connective tissue for any LLM model handling, with a few key projects in mind. 

on the todo list is clear up the issues ive already made, feel free to raise any issues you see but at this point this will be a very iterative project to connect a lot of different method together and will require a few refactors along the way. so we'll prototype fast and do our best not to leave too much of a mess along the way. keeping the project dynamic and modular are the practices most key.

the use of subprocess as the runtime allows the resources of this project to be defined somewhat and will need to be considered in one of the early refactors, with the recent addition of bash to windows natively this really opens up the ability to smash this code out in a single structure granted the end user has a bash terminal looking at you file name syntaxes"/" "\"

this project is about to ballon. honestly, forgive me.



How to run:
```bash
git clone https://www.github.com/taylor-ennen/DINO-Digestion
cd DINO-Digestion/
git clone https://www.github.com/ggerganov/llama.cpp

python3 -m pip install -r requirements.txt
```

Now that the requirements are fulfilled you can run -help to see the options for the download_and_convert_to_gguf.py script:
```bash
python3 download_and_convert_to_gguf.py -h
```

Finally, if you just want a quick small model to test with, you can run (you'll need to have room for the model and conversion into gguf format, so about 20GB of space free on your machine including the libraries in the requirements for llama.cpp):
```bash
python3 download_and_convert_to_gguf.py -r microsoft/phi-2 -o f16
```
There is also a colab with the command needed to utilize this in a notebook for examples sake:
https://colab.research.google.com/drive/14SIfW93N5cAVfLDX3l23Xjm5btPWR8SV#scrollTo=f68bYcaYDf1S
