=== USAGE ===
To execute the program, simply type `python3 run.py` into a terminal open in
this directory. This will perform data preparation on `data2021.student.csv`,
run the validation/model selection process, train the final model, make the
final predictions, and save those to `results.csv`. User-adjustable variables
are listed amongst the data preparation code where the apply and at the begining
for those related to the models.

=== PROBLEMS ===
I am not aware of any problems with the code as is. There is code for displaying
the correlation matrix and code for adding PCA to the pipeline that has been
commented out for future personal reference. That code should not cause any
errors if correctly uncommented, but using PCA will decrease accuracy with my
current implementation.