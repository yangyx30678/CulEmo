# How to use this repository
Once you perfectly understand this guidance, you can remove this file. 
If you need to read it again, see https://github.com/isct-lrlab/xxG-Student-Template.

## Purpose of using this repository
* As your **[lab notebook](https://en.wikipedia.org/wiki/Lab_notebook)**
  * **To assure originality and reproducibility of your research**
* Sharing your code to other lab members to obtain advice from them
* Efficiently deploy your code both in lab servers and TSUBAME
* Sharing coding techniques and tips with other lab members
* Transferring your research effort and results when you leave the lab to the supervisor or new students who continue or extend your work

## Initial instructions
1. Update README.md with your information by filling fields marked by `(( ))`. If you do not have either your TSUBAME or Overleafe acocunt, create one.
2. Create your Overleaf account and tell it to funakoshi (You will be provided an Overleaf project for your research survey and reports)
3. Add the links to your GitHub research repository (this repository) and to your Overleaf project in the direct message (DM) with funakoshi of the lab slack workspace 

## General instructions
* Do not put your confidential information in this repo
  * E.g., if you work in a collaboration under NDA, use other repository for the collaboration work
  * This repository is private but open for lab members
* Do not put heavy data and intemediate model snapshots in this repo
  * However, you should put your final model training weights when you conculde your research
* Use the inside of the repo as you like
  * How to organize the directory tree and branch tree is up to you
  * However, keep clear and easy-to-follow structures for others
* Use meaningful names for files and directories

## Tips
* Put a readme file for every folder (in English as much as possible but Japanese is also fine)
  * Describe what you do or what files are in the folder
  * If you did some manual work such as data preprocessing in the folder, leave the note of the procedure so that you (and others) can easily reproduce the work again
* Write a shell script for every experiment
  * Leave all commandline parameters in the shell script so that ANYBODY can re-run the experiment (i.e., can reproduce the results) again
* Leave raw experimental results log files (not only calculated metrics scores) so that you can conduct statistical tests later.
