import bilsm_crf_model
import process_data
import numpy as np

# EPOCHS = 3
# model, (train_x, train_y) = bilsm_crf_model.create_model()
# model.fit(train_x[:10000], train_y[:10000],batch_size=64,epochs=EPOCHS)
# model.save('model/crf.h5')

EPOCHS = 50
model, (train_x, train_y) = bilsm_crf_model.create_model()
model.fit(train_x, train_y,batch_size=64,epochs=EPOCHS)
model.save('model/crf.h5')
