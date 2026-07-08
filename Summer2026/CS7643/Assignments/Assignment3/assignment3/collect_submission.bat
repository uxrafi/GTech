@echo off
if exist assignment3_nlp_submission.zip del /F /Q assignment3_nlp_submission.zip
tar -a -c -f assignment3_nlp_submission.zip models Machine_Translation.ipynb