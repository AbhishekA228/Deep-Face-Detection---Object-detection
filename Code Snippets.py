Stock Code Snippets:
    
#Avoid OOM errors by setting GPU Memory Consumption Growth
#CODE:
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
    
#Code for Train,Test,Split Dataset
for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data',folder,'images')):
        filename=file.split('.')[0]+'.json'
        existing_filepath=os.path.join('data','labels',filename)
        if os.path.exists(existing_filepath):
            new_filepath=os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath,new_filepath)
            #changes labels with respect to their corresponding file location in the test,train,val folders
            
    
