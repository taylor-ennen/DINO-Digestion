So this project aims to be the connective tissue for any LLM model handling, with a few key projects in mind. 

on the todo list is clear up the issues ive already made, feel free to raise any issues you see but at this point this will be a very iterative project to connect a lot of different method together and will require a few refactors along the way. so we'll prototype fast and do our best not to leave too much of a mess along the way. keeping the project dynamic and modular are the practices most key.

the use of subprocess as the runtime allows the resources of this project to be defined somewhat and will need to be considered in one of the early refactors, with the recent addition of bash to windows natively this really opens up the ability to smash this code out in a single structure granted the end user has a bash terminal looking at you file name syntaxes"/" "\"

this project is about to ballon. honestly, forgive me.



How to run:

git clone https://www.github.com/taylor-ennen/DINO-Digestion
cd DINO-Digestion/
git clone https://www.github.com/ggerganov/llama.cpp

python3 -m pip install -r requirements.txt

python3 download_and_convert_to_gguf.py -r microsoft/phi-2 -o f16

