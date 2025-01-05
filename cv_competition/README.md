Use my_model_wrapper.py for your "Efficient model", see an example in my model_wrapper_quantized.py for a quantized model.  <br>

To validate the model 'run_rests.py' will be executed upon git push. <br>

Model will be evaluated for:
> inference speed <br>
> weighst size  <br>
> MSE and JACCARD metircs <br>

All evaluation will be pefrormed on CPU. <br>

To validate model performance locally - downlad 'validation_images' data from <a href="https://drive.google.com/file/d/1PzqONtbTwst_2SxgCQ5YenAuKHvkycV6/view?usp=drive_link">here</a>. To speficy images location use config.yaml. <br>

Model pefromance results will be stored in 'results.json'.

For this competition you need to push this folder only as a repository.
