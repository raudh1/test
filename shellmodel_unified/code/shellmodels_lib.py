#------------------SHELL MODELS LIBRARY----------------------#
from libraries import *

##################----------------DEVICE/CPU/GPU---------###################
device = th.device('cpu')
if th.cuda.is_available():
    th.backends.cudnn.benchmark = True
    device = th.device('cuda')
    print("device = ", device)
print("device:",device) 

######################--------LOAD DATA-------------------################

def load_data(path,filename,nshells=12,sampling=1
,difference=False,normalize01=True,normalize11=False,verbose=False):
    """ This function loads the dataset of a given path 
    and does the preprocessing. 
    You can select:
    -nshells
    -sampling
    -difference  : instead of x_n it uses x_n'=x_n-x_{n-1} -- it should shows better performances
    -normalize01 : to have data that are in range (0,1)
    -normalize11 : to have data that are in range (-1,1)
    -verbose

    """
    alldata = np.load(path+filename)[2:-2]
    if verbose:
        print("Original data shape : ", alldata.shape)

    data = np.expand_dims(alldata.T, axis=1)
    data = data[:,:,:nshells]
    data = data[::sampling]
    data = np.abs(data)
    L = len(data)
    print("After subsampling : ", data.shape )
    if difference:
        data = data[1:] - data[:-1]  # x_n-x_{n-1}

    if normalize01:
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        if normalize11:
            data = data * 2 - 1    

    if normalize11:
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        data = data * 2 - 1    
    
   
    return data

######################--------SPLIT DATASET-------------------################

def split_data(data,start=0):
    """
        To use after load_data
        split the dataset in three parts : 

        - train  
        - valid
        - test

        start :   begin of training set
        
        train :   from start            ---> start+train
        val   :   from start+train      ---> start+train+val
        test  :   from start+train+val  ---> start+train+val+test

        return train/val/test
    """
    train=len(data)*8//10
    val=len(data)*15//100
    test=len(data)*5//10

    train_data = th.FloatTensor((data[start:start+train,:,:]))#.to(device)
    valid_data = th.FloatTensor((data[start+train:start+train+val,:,:]))#.to(device)
    test_data = th.FloatTensor((data[start+train+val:start+train+val+test,:,:]))#.to(device)
    print("train data: ", train_data.shape)
    print("valid data: ", valid_data.shape)
    print("test data: ", test_data.shape)

    return train_data, valid_data, test_data

def struct_func(u_n,order,h,f,title,save=False,folder=''):
    """
        Compute the Structure function given |U_n| :
            S_q= mean ( |U_n(t)|^q ) 
        In which mean is the time average.
    """
    print("evaluate structure function S_",order)
    Sq=np.mean(np.power(u_n,order),0)
        
    if save:
        np.save('./'+str(folder)+'/S_'+str(order)
        +'_h_'+str(h)+'_f_'+str(f)+str(title),Sq.T)

        print('saved S_'+str(order)
        +'_h_'+str(h)+'_f_'+str(f)+str(title))
    return Sq.T

######################--------BATCH-------------------#################
def batchify(data, indices, history, future, model):
    bs = len(indices)
    S = data.shape[-1]


    if isinstance(model, Sequence):
        
        outX = th.zeros(history, bs, S)
        outY = th.zeros(future, bs, S)
        for i in range(bs):
            start = indices[i]
            outX[:, i:i + 1, :] = data[start:start + history].to(device)
            outY[:, i:i + 1, :] = data[start + history:start + history + future].to(device)
        return outX, outY

    if isinstance(model, MLP):
        
        outX = th.zeros(bs, history * S,batch_first=True)
        outY = th.zeros(bs, future * S,batch_first=True)
        for i in range(bs):
            start = indices[i]
            outX[i, :] = data[start:start + history].flatten().to(device)
            outY[i, :] = data[start + history:start + history + future].flatten().to(device)
        return outX, outY

#-----------------------------------------------------TRAIN FUNCTIONS-----------------------------------------------------------------------#

class TrainFunction:
    def __init__(self, 
                model,
                optimizer,
                train,
                valid,
                history=100,
                future=10,
                nepochs=10,
                bsize=100,
                nprints=10,
                save=False
                ):

        self.model=model
        self.optimizer=optimizer
        self.train=train
        self.valid=valid
        self.history=history
        self.future=future
        self.nepochs=nepochs
        self.nprints=nprints
        self.save=save

    def train(self):
        pass

class STDtraining(TrainFunction):


    def __init__(self, model, optimizer, train, valid, history=100, future=10, nepochs=10, bsize=100, nprints=10, save=False):
        super().__init__(model, optimizer, train, valid, history, future, nepochs, bsize, nprints, save)


    def get_param(self,model,optimizer):

        #------------------------------------------chose model---------------------------------------#
        if isinstance(model, Sequence):
            model_name='lstm'
        elif isinstance(model,MLP):
            model_name='mlp'
        
        #------------------------------------------chose optimizer-----------------------------------#

        if isinstance(optimizer, th.optim.Adam):
            optimizer_name='adam'
        elif isinstance(optimizer, th.optim.LBFGS):
            optimizer_name='lbfgs'

        return model_name,optimizer_name

    def train(self,model, optimizer, train, valid, nepochs=10, nprints=10,save=False,folder=''):
        """
        Basic training

        The training sequence is processed as a long sequence. 
        For each time step       
        y_{t+1} = model(x_t,h_t)
        
        """
        train_loss = []
        valid_loss = []
        i = 0
        min_perf=900000          # param used to find (and save) min validation loss model
        criterion = nn.MSELoss()
        


        model_name,optimizer_name=self.get_param(model,optimizer)

        if optimizer_name=='adam':
            while i < nepochs:
                model.train()
                optimizer.zero_grad()
                if model_name=='lstm':
                    out, (h, c) = model(train[:-1])
                    loss = criterion(out, train[1:])
                elif model_name=='mlp':
                    out = model(train[:-1])
                    loss = criterion(out, train[1:])
                        
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                # validation
                with th.no_grad():
                    model.eval()
                    if model_name=='lstm':
                        pred, _ = model(valid[:-1])
                        validloss = criterion(pred, valid[1:])
                    elif model_name=='mlp':
                        pred = model(valid[:-1])
                        validloss = criterion(pred, valid[1:])
                    
                    
                    valid_loss.append(validloss.item())
                i += 1
                if i % (nepochs // nprints) == 0:
                    print(i, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])
                
                if validloss.item() < min_perf :
                    min_perf = validloss.item()
                    th.save(model.state_dict(), './'+str(folder)+'/models/epochs_'+str(nepochs))
                    bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/epochs_'+str(nepochs)))
            model=bestmodel

            return train_loss, valid_loss

        if optimizer_name=='lbfgs':
            optim = th.optim.LBFGS(model.parameters())                
            train_loss = []
            valid_loss = []
            e = 0
            min_perf=900000          # param used to find (and save) min validation loss model

            criterion = nn.MSELoss()
            while e < nepochs:
                def closure(): 
                    optim.zero_grad()
                    if model_name=='lstm':
                        out, (h, c) = model(train[:-1].to(device))
                        loss = criterion(out.to(device), train[1:].to(device))
                    if model_name=='mlp':
                        out = model(train[:-1])
                        loss = criterion(out, train[1:])
                    loss.backward()
                    return loss
                model.train()
                loss_epoch = optim.step(closure)
                train_loss.append(loss_epoch.item())    
                # validation
                with th.no_grad():
                    model.eval()
                    if model_name=='lstm':
                        pred, _ = model(valid[:-1])
                        validloss = criterion(pred, valid[1:])
                    if model_name=='mlp':
                        pred = model(valid[:-1])
                        validloss = criterion(pred, valid[1:])

                    valid_loss.append(validloss.item())
                e += 1
                if e % (nepochs // nprints) == 0:
                    print(e, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])
                if validloss.item() < min_perf :
                    min_perf = validloss.item()
                    th.save(model.state_dict(), './results_LBFGS/models/epochs_'+str(nepochs))
                    bestmodel=model.load_state_dict(th.load('./results_LBFGS/models/epochs_'+str(nepochs)))
            model=bestmodel
            return train_loss, valid_loss


class FORtraining(TrainFunction):

    def __init__(self, model, optimizer, train, valid, history=100, future=10,
     nepochs=10, bsize=100, nprints=10, save=False):
        super().__init__(model, optimizer, train, valid, history, future, nepochs, bsize, nprints, save)


    def get_param(self,model,optimizer):

            #------------------------------------------chose model---------------------------------------#
            if isinstance(model, Sequence):
                model_name='lstm'
            elif isinstance(model,MLP):
                model_name='mlp'
            
            #------------------------------------------chose optimizer-----------------------------------#

            if isinstance(optimizer, th.optim.Adam):
                optimizer_name='adam'
            elif isinstance(optimizer, th.optim.LBFGS):
                optimizer_name='lbfgs'

            return model_name,optimizer_name

    def train(self,model,
                    optimizer,        
                    train,
                    valid,
                    history=100,
                    future=10,
                    nepochs=10,
                    bsize=100,
                    nprints=10,
                    patience=3,
                    save=True,
                    folder=''):


        """## With forecast training as fine tuning 

        forecast training takes as parameters: 
        - history: $h$,  the time span read by the LSTM
        $$
        \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
        $$
        - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
        $$
        \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
        $$

        """
        bestmodel=model
        #model=model.to(device)
        ## Data splitting
        # Train
        L = len(train)
        nex = (L - future - history)
        nbatch = nex // bsize
        # Valid
        LV = len(valid)
        nexV = (LV - future - history)
        nbatchV = nexV // bsize
        print("n batch valid loss ",nbatchV)
        # Random split of the train
        indices = np.arange(nex)
        indicesV = np.arange(nexV)
        np.random.shuffle(indices)
        # Init.
        #optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)

        train_loss = []
        valid_loss = []
        e = 0
        counter=0
        loss_lag=42
        min_perf=900000          # param used to find (and save) min validation loss model
        criterion = nn.MSELoss()

        model_name,optimizer_name=self.get_param(model,optimizer)


        if optimizer_name=='lbfgs':
            
            optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)

            while e < nepochs:
                global eloss 
                eloss = 0
                evalloss=0
                model.train()
                for i in range(nbatch):
                    optimizer.zero_grad()
                    bidx = indices[i * bsize:min((i + 1) * bsize, L)]
                    inputs, refs = batchify(train, bidx, history, future,model)
                       
                    def closure():
                        optimizer.zero_grad()
                        h0, c0 = model.get_init(bsize)
                        outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
                        loss = criterion(outputs, refs.to(device))
                        loss.backward()
                        global eloss
                        eloss += loss
                        return eloss
        
                loss_b=optimizer.step(closure)
                train_loss.append(loss_b.item())
            
                
                if (np.abs(loss_lag-loss_b.item())<1e-5):
                    counter+=1
                    #print("counter=",counter)
                    #print("loss_lag=",loss_lag)
                    #print("loss_b =",loss_b)
                    #print("patience=",patience)
                    if (counter==patience):
                        print("loss is not decreasing anymore")
                        if e>100:
                            print("training stopped at epoch ",e," and model saved")
                            break
                        else:
                            print("training stopped at epoch ",e," and model not saved")
                            sys.exit()
                else:
                    counter=0
                    loss_lag=loss_b.item()
                    
                #validation
                with th.no_grad():
                    model.eval()
                    for i in range(nbatchV):
                        #print("!!! in batch !!")
                        bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
                        inputs, refs = batchify(valid, bidx, history, future)
                        h0, c0 = model.get_init(bsize)
                        outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
                        validloss = criterion(outputs, refs.to(device))
                        #print(" gave value to validloss")
                    valid_loss.append(validloss.item())
                e += 1
                if e % (nepochs // nprints) == 0:
                    print(e, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])
                if save:
                    if validloss.item() < min_perf :
                        min_perf = validloss.item()
                        name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
                        th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
                        bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/'+name))
            model=bestmodel
            return train_loss, valid_loss

        if optimizer_name=='adam':

            criterion = nn.MSELoss()
            while e < nepochs:
                model.train()
                eloss = 0
                for i in range(nbatch):
                    optimizer.zero_grad()
                    bidx = indices[i * bsize:min((i + 1) * bsize, L)]
                    inputs, refs = batchify(train, bidx, history, future,model)

                    if model_name=='lstm':
                        h0, c0 = model.get_init(bsize)
                        outputs, (h, c) = model.generate(inputs, future, h0, c0)
                    elif model_name=='mlp':
                        outputs = model.forward(inputs.to(device))
                    
                    loss = criterion(outputs, refs)
                    loss.backward()
                    optimizer.step()
                    eloss += loss.item()
                    #print(eloss)    
                
                train_loss.append(eloss)

                if (np.abs(loss_lag-loss_b.item())<1e-5):
                    counter+=1
                    #print("counter=",counter)
                    #print("loss_lag=",loss_lag)
                    #print("loss_b =",loss_b)
                    #print("patience=",patience)
                    if (counter==patience):
                        print("loss is not decreasing anymore")
                        if e>100:
                            print("training stopped at epoch ",e," and model saved")
                            break
                        else:
                            print("training stopped at epoch ",e," and model not saved")
                            sys.exit()
                else:
                    counter=0
                    loss_lag=loss_b.item()



                # validation
                with th.no_grad():
                    model.eval()
                    for i in range(nbatchV):
                        bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
                        inputs, refs = batchify(valid, bidx, history, future,model)

                        if model_name=='lstm':
                            h0, c0 = model.get_init(bsize)
                            outputs, (h, c) = model.generate(inputs, future, h0, c0)
                        elif model_name=='mlp':    
                            outputs = model.forward(inputs.to(device))    

                        validloss = criterion(outputs, refs)
                    valid_loss.append(validloss.item())
                e += 1
                if e % (nepochs // nprints) == 0:
                    print(e, "train loss", train_loss[-1], "valid loss",
                        valid_loss[-1])
                if save:
                    if validloss.item() < min_perf :
                        min_perf = validloss.item()
                        name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
                        th.save(model.state_dict(), './results/models/'+name)
                
            return train_loss, valid_loss    
            













# ######################--------FORECAST TRAIN----------#################
# def forecast_train(model,
#                    optimizer,
#                    train,
#                    valid,
#                    history=100,
#                    future=10,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    save=False):

#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     """
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     train_loss = []
#     valid_loss = []
#     e = 0
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while e < nepochs:
#         model.train()
#         eloss = 0
#         for i in range(nbatch):
#             optimizer.zero_grad()
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)
#             h0, c0 = model.get_init(bsize)
#             outputs, (h, c) = model.generate(inputs, future, h0, c0)
#             loss = criterion(outputs, refs)
#             loss.backward()
#             optimizer.step()
#             eloss += loss.item()
#             #print(eloss)    
#         train_loss.append(eloss)

#         # validation
#         with th.no_grad():
#             model.eval()
#             for i in range(nbatchV):
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs, future, h0, c0)
#                 validloss = criterion(outputs, refs)
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                   valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
#                 th.save(model.state_dict(), './results/models/'+name)
         
#     return train_loss, valid_loss

# def forecast_train_LBFGS_p(model,        
#                    train,
#                    valid,
#                    history=100,
#                    future=10,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    patience=3,
#                    save=True,
#                    folder=''):


#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     """
#     bestmodel=model
#     #model=model.to(device)
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     print("n batch valid loss ",nbatchV)
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)

#     train_loss = []
#     valid_loss = []
#     e = 0
#     counter=0
#     loss_lag=42
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while e < nepochs:
#         global eloss 
#         eloss = 0
#         evalloss=0
#         model.train()
#         for i in range(nbatch):
#             optimizer.zero_grad()
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)

#             def closure():
#                 optimizer.zero_grad()
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
#                 loss = criterion(outputs, refs.to(device))
#                 loss.backward()
#                 global eloss
#                 eloss += loss
#                 return eloss
  
#             loss_b=optimizer.step(closure)
#             train_loss.append(loss_b.item())
    
        
#         if (np.abs(loss_lag-loss_b.item())<1e-5):
#             counter+=1
#             #print("counter=",counter)
#             #print("loss_lag=",loss_lag)
#             #print("loss_b =",loss_b)
#             #print("patience=",patience)
#             if (counter==patience):
#                 print("loss is not decreasing anymore")
#                 if e>100:
#                     print("training stopped at epoch ",e," and model saved")
#                     break
#                 else:
#                     print("training stopped at epoch ",e," and model not saved")
#                     sys.exit()
#         else:
#             counter=0
#         loss_lag=loss_b.item()
            
#          #validation
#         with th.no_grad():
#             model.eval()
#             for i in range(nbatchV):
#                 #print("!!! in batch !!")
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
#                 validloss = criterion(outputs, refs.to(device))
#                 #print(" gave value to validloss")
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                 valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
#                 th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
#                 bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/'+name))
#     model=bestmodel
#     return train_loss, valid_loss


# ######################--------FORECAST TRAIN MLP ----------#################
# def forecast_train_MLP(model,
#                    optimizer,
#                    train,
#                    valid,
#                    history=1,
#                    future=25,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    patience=50,
#                    save=False,
#                    folder=''):




#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     """
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     train_loss = []
#     valid_loss = []
#     e = 0
#     loss_lag=42
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while e < nepochs:
#         model.train()
#         eloss = 0
#         for i in range(nbatch):
#             optimizer.zero_grad()
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)
            
            
#             outputs = model.generate(inputs.to(device), future).to(device)
#             loss = criterion(outputs, refs.to(device))
#             loss.backward()
#             optimizer.step()
#             eloss += loss.item()
#             #print(eloss)    
#         train_loss.append(eloss)

#         if (np.abs(loss_lag-train_loss[e])<1e-5):
#             counter+=1
#             #print("counter=",counter)
#             #print("loss_lag=",loss_lag)
#             #print("loss_b =",loss_b)
#             #print("patience=",patience)
#             if (counter==patience):
#                 print("loss is not decreasing anymore")
#                 if e>100:
#                     print("training stopped at epoch ",e," and model saved")
#                     break
#                 else:
#                     print("training stopped at epoch ",e," and model not saved")
#                     exit
#         else:
#             counter=0
#         loss_lag=train_loss[e]




#         # validation
#         with th.no_grad():
#             model.eval()
#             for i in range(nbatchV):
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 outputs = model.generate(inputs.to(device), future)
#                 validloss = criterion(outputs, refs.to(device))
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                   valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
#                 th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
         
#     return train_loss, valid_loss

# def forecast_train_MLP_LBFGS_p(model,
#                    train,
#                    valid,
#                    history=100,
#                    future=10,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    patience=50,
#                    save=False,
#                    folder=''):


    

#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     """
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     optimizer = th.optim.LBFGS(model.parameters(),lr=0.001)

#     train_loss = []
#     valid_loss = []
#     e = 0
#     loss_lag=42
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()

#     while e < nepochs:
#         global eloss
#         eloss = 0
#         for i in range(nbatch):
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)
            
#             def closure():
#                 optimizer.zero_grad()
#                 outputs = model.generate(inputs.to(device), future).to(device)
#                 loss = criterion(outputs, refs.to(device))
#                 loss.backward()
#                 #optimizer.step()
#                 global eloss
#                 eloss += loss.item()
#                 return eloss
#                 #print(eloss)    
#                 #train_loss.append(eloss)
#         model.train()

#         loss_b=optimizer.step(closure)
#         train_loss.append(loss_b)


#         if (np.abs(loss_lag-train_loss[e])<1e-5):
#             counter+=1
#             #print("counter=",counter)
#             #print("loss_lag=",loss_lag)
#             #print("loss_b =",loss_b)
#             #print("patience=",patience)
#             if (counter==patience):
#                 print("loss is not decreasing anymore")
#                 if e>50:
#                     print("training stopped at epoch ",e," and model saved")
#                     break
#                 else:
#                     print("training stopped at epoch ",e," and model not saved")
#                     exit()
#         else:
#             counter=0
#         loss_lag=loss_b
#         if math.isnan(loss_b):
#             print("Loss is nan, training stopped at epoch",e)
#             exit()



#         # validation
#         with th.no_grad():
#             model.eval()
#             for i in range(nbatchV):
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 outputs = model.generate(inputs.to(device), future)
#                 validloss = criterion(outputs, refs.to(device))
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                   valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
#                 th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
         
#     return train_loss, valid_loss

# ######################--STANDARD TRAIN ADAM -#################

# def train(model, optimizer, train, valid, nepochs=10, nprints=10,save=False,folder=''):
#     """
#     Basic training

#     The training sequence is processed as a long sequence. 
#     For each time step       
#     y_{t+1} = model(x_t,h_t)
    
#     """

#     train_loss = []
#     valid_loss = []
#     i = 0
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while i < nepochs:
#         model.train()
#         optimizer.zero_grad()
#         out, (h, c) = model(train[:-1])
#         loss = criterion(out, train[1:])
#         loss.backward()
#         optimizer.step()
#         train_loss.append(loss.item())
#         # validation
#         with th.no_grad():
#             model.eval()
#             pred, _ = model(valid[:-1])
#             validloss = criterion(pred, valid[1:])
#             valid_loss.append(validloss.item())
#         i += 1
#         if i % (nepochs // nprints) == 0:
#             print(i, "train loss", train_loss[-1], "valid loss",
#                   valid_loss[-1])
        
#         if validloss.item() < min_perf :
#             min_perf = validloss.item()
#             th.save(model.state_dict(), './'+str(folder)+'/models/epochs_'+str(nepochs))
#             bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/epochs_'+str(nepochs)))
#     model=bestmodel

#     return train_loss, valid_loss

# def train_LBFGS(model, train, valid, nepochs=10, nprints=10):
#     optim = th.optim.LBFGS(model.parameters())
#     model=model.to(device)
#     train_loss = []
#     valid_loss = []
#     e = 0
#     min_perf=900000          # param used to find (and save) min validation loss model

#     criterion = nn.MSELoss()
#     while e < nepochs:
#         def closure(): 
#             optim.zero_grad()
#             out, (h, c) = model(train[:-1].to(device))
#             loss = criterion(out.to(device), train[1:].to(device))
#             loss.backward()
#             return loss
#         model.train()
#         loss_epoch = optim.step(closure)
#         train_loss.append(loss_epoch.item())    
#         # validation
#         with th.no_grad():
#             model.eval()
#             pred, _ = model(valid[:-1])
#             validloss = criterion(pred, valid[1:])
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                   valid_loss[-1])
#         if validloss.item() < min_perf :
#             min_perf = validloss.item()
#             th.save(model.state_dict(), './results_LBFGS/models/epochs_'+str(nepochs))
#             bestmodel=model.load_state_dict(th.load('./results_LBFGS/models/epochs_'+str(nepochs)))
#     model=bestmodel
#     return train_loss, valid_loss

# ######################--------TRAIN-MLP---------------################
# def train_MLP(model, optimizer, train, valid, nepochs=10, nprints=10,save=False,folder=''):
#     """
#     Basic training

#     The training sequence is processed as a long sequence. 
#     For each time step       
#     y_{t+1} = model(x_t,h_t)
    
#     """

#     train_loss = []
#     valid_loss = []
#     i = 0
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while i < nepochs:
#         model.train()
#         optimizer.zero_grad()
#         out = model(train[:-1])
#         loss = criterion(out, train[1:])
#         loss.backward()
#         optimizer.step()
#         train_loss.append(loss.item())
#         # validation
#         with th.no_grad():
#             model.eval()
#             pred = model(valid[:-1])
#             validloss = criterion(pred, valid[1:])
#             valid_loss.append(validloss.item())
#         i += 1
#         if i % (nepochs // nprints) == 0:
#             print(i, "train loss", train_loss[-1], "valid loss",
#                   valid_loss[-1])
        
#         if validloss.item() < min_perf :
#             min_perf = validloss.item()
#             th.save(model.state_dict(), './'+str(folder)+'/models/epochs_'+str(nepochs))
#             bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/epochs_'+str(nepochs)))
#     model=bestmodel

#     return train_loss, valid_loss











# ######################--FORECAST TRAIN ADAM -#################

# def forecast_train(model,
#                    optimizer,
#                    train,
#                    valid,
#                    history=100,
#                    future=10,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    save=False):




#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     """
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     train_loss = []
#     valid_loss = []
#     e = 0
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while e < nepochs:
#         model.train()
#         eloss = 0
#         for i in range(nbatch):
#             optimizer.zero_grad()
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)
#             h0, c0 = model.get_init(bsize)
#             outputs, (h, c) = model.generate(inputs, future, h0, c0)
#             loss = criterion(outputs, refs)
#             loss.backward()
#             optimizer.step()
#             eloss += loss.item()
#             #print(eloss)    
#         train_loss.append(eloss)

#         # validation
#         with th.no_grad():
#             model.eval()
#             for i in range(nbatchV):
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs, future, h0, c0)
#                 validloss = criterion(outputs, refs)
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                   valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
#                 th.save(model.state_dict(), './results/models/'+name)
         
#     return train_loss, valid_loss


# ######################--FORECAST TRAIN LBFGS patience-#################

# def forecast_train_LBFGS_p(model,        
#                    train,
#                    valid,
#                    history=100,
#                    future=10,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    patience=3,
#                    save=True,
#                    folder=''):


#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     """
#     bestmodel=model
#     #model=model.to(device)
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     print("n batch valid loss ",nbatchV)
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)

#     train_loss = []
#     valid_loss = []
#     e = 0
#     counter=0
#     loss_lag=42
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while e < nepochs:
#         global eloss 
#         eloss = 0
#         evalloss=0
#         for i in range(nbatch):
#             #optimizer.zero_grad()
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)

#             def closure():
#                 optimizer.zero_grad()
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
#                 loss = criterion(outputs, refs.to(device))
#                 loss.backward()
#                 global eloss
#                 eloss += loss
#                 return eloss
#         model.train()
        
#         loss_b=optimizer.step(closure)
#         train_loss.append(loss_b.item())
        
        
#         if (np.abs(loss_lag-loss_b.item())<1e-5):
#             counter+=1
#             #print("counter=",counter)
#             #print("loss_lag=",loss_lag)
#             #print("loss_b =",loss_b)
#             #print("patience=",patience)
#             if (counter==patience):
#                 print("loss is not decreasing anymore")
#                 if e>100:
#                     print("training stopped at epoch ",e," and model saved")
#                     break
#                 else:
#                     print("training stopped at epoch ",e," and model not saved")
#                     exit
#         else:
#             counter=0
#         loss_lag=loss_b.item()
            
#          #validation
#         with th.no_grad():
#             model.eval()
#             for i in range(nbatchV):
#                 print("!!! in batch !!")
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
#                 validloss = criterion(outputs, refs.to(device))
#                 print(" gave value to validloss")
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                 valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)
#                 th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
#                 bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/'+name))
#     model=bestmodel
#     return train_loss, valid_loss

# def forecast_train_LBFGS_p_last_point(model,        
#                    train,
#                    valid,
#                    history=100,
#                    future=10,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    patience=3,
#                    s=10,
#                    save=True,
#                    folder='',
#                    run=1
#                    ):
#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     After s epochs the model learn only using the last generated point, namely it evaluate the
#     loss function only using the last point.

    
    
#     """
#     bestmodel=model
#     #model=model.to(device)
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     print("n batch valid loss ",nbatchV)
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)

#     train_loss = []
#     valid_loss = []
#     e = 0
#     counter=0
#     loss_lag=42
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while e < nepochs:
#         global eloss 
#         eloss = 0
#         evalloss=0
#         for i in range(nbatch):
#             #optimizer.zero_grad()
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)

#             def closure():
#                 optimizer.zero_grad()
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
#                 if e < s:
#                     loss = criterion(outputs, refs.to(device))
#                 else:
#                     loss = criterion(outputs[:, -1], refs[:, -1].to(device))
#                 loss.backward()
#                 global eloss
#                 eloss += loss
#                 return eloss
#         model.train()
        
#         loss_b=optimizer.step(closure)
#         train_loss.append(loss_b.item())
        
        
#         if (np.abs(loss_lag-loss_b.item())<1e-5):
#             counter+=1
#             #print("counter=",counter)
#             #print("loss_lag=",loss_lag)
#             #print("loss_b =",loss_b)
#             #print("patience=",patience)
#             if (counter==patience):
#                 print("loss is not decreasing anymore")
#                 if e>100:
#                     print("training stopped at epoch ",e," and model saved")
#                     break
#                 else:
#                     print("training stopped at epoch ",e," and model not saved")
#                     exit
#         else:
#             counter=0
#         loss_lag=loss_b.item()
            
#          #validation
#         with th.no_grad():
#             model.eval()
#             for i in range(nbatchV):
#                 #print("!!! in batch !!")
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
#                 if e < s: # s is the time we want to use all the traj to compute the loss, after s epochs the model evaluates the loss only on the last point
#                     validloss = criterion(outputs, refs.to(device))
#                 else:
#                     if (e==s):
#                         print(" Changing to evaluate loss only with the last point  ")
#                     validloss = criterion(outputs[:, -1], refs[:, -1].to(device))
#                 #print(" gave value to validloss")
#             valid_loss.append(validloss.item())
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                 valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)+'_run_'+str(run)
#                 th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
#                 bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/'+name))
#     model=bestmodel
#     return train_loss, valid_loss

# def forecast_train_LBFGS_p_all_point(model,        
#                    train,
#                    valid,
#                    history=100,
#                    future=10,
#                    nepochs=10,
#                    bsize=100,
#                    nprints=10,
#                    patience=3,
#                    s=10,
#                    save=True,
#                    folder='',
#                    run=1
#                    ):
#     """## With forecast training as fine tuning 

#     forecast training takes as parameters: 
#     - history: $h$,  the time span read by the LSTM
#     $$
#     \text{for }t \leq h,\ y_{t+1} = model(x_t,h_t)
#     $$
#     - future: $f$, after the LSTM has read h inputs, the LSTM generates the $f$ next outputs
#     $$
#     \text{for }h \leq t \leq h+f,\ y_{t+1} = model(y_t,h_t)
#     $$

#     After s epochs the model learn only using the last generated point, namely it evaluate the
#     loss function only using the last point.

    
    
#     """
#     bestmodel=model
#     #model=model.to(device)
#     ## Data splitting
#     # Train
#     L = len(train)
#     nex = (L - future - history)
#     nbatch = nex // bsize
#     # Valid
#     LV = len(valid)
#     nexV = (LV - future - history)
#     nbatchV = nexV // bsize
#     print("n batch valid loss ",nbatchV)
#     # Random split of the train
#     indices = np.arange(nex)
#     indicesV = np.arange(nexV)
#     np.random.shuffle(indices)
#     # Init.
#     optimizer = th.optim.LBFGS(model.parameters(),lr=0.1)

#     train_loss = []
#     valid_loss = []
#     e = 0
#     counter=0
#     loss_lag=42
#     min_perf=900000          # param used to find (and save) min validation loss model
#     criterion = nn.MSELoss()
#     while e < nepochs:
#         global eloss 
#         eloss = 0
#         evalloss=0
#         for i in range(nbatch):
#             bidx = indices[i * bsize:min((i + 1) * bsize, L)]
#             inputs, refs = batchify(train, bidx, history, future)

#             def closure():
#                 optimizer.zero_grad()
#                 h0, c0 = model.get_init(bsize)
#                 outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0)
                
#                 # Calcolo la loss pointwise per ogni punto di dati e aggiorno i pesi del modello
#                 for j in range(outputs.size(1)):
#                     if e < s:
#                         loss = criterion(outputs, refs.to(device))
#                     else:
#                         loss = criterion(outputs[:, j, :], refs[:, j, :].to(device))
#                     loss.backward(retain_graph=True)
#                     optimizer.step()
#                     optimizer.zero_grad()
                
#                 # Ottengo la loss totale come un float
#                 global eloss
#                 eloss += loss.item()
#                 return eloss

#         # Eseguo la fase di training
#         model.train()
#         loss_b = optimizer.step(closure)
#         train_loss.append(loss_b)

            
#         if (np.abs(loss_lag-loss_b.item())<1e-5):
#             counter+=1
#             #print("counter=",counter)
#             #print("loss_lag=",loss_lag)
#             #print("loss_b =",loss_b)
#             #print("patience=",patience)
#             if (counter==patience):
#                 print("loss is not decreasing anymore")
#                 if e>100:
#                     print("training stopped at epoch ",e," and model saved")
#                     break
#                 else:
#                     print("training stopped at epoch ",e," and model not saved")
#                     exit
#         else:
#             counter=0
#         loss_lag=loss_b.item()
            
#         #validation
#         with th.no_grad():
#             model.eval()



#             for i in range(nbatchV):
#                 #print("!!! in batch !!")
#                 bidx = indicesV[i * bsize:min((i + 1) * bsize, LV)]
#                 inputs, refs = batchify(valid, bidx, history, future)
#                 def closure():
#                     h0, c0 = model.get_init(bsize)
#                     outputs, (h, c) = model.generate(inputs.to(device), future, h0, c0) 
#                     # Calcolo la loss pointwise per ogni punto di dati e aggiorno i pesi del modello
#                     for j in range(outputs.size(1)):
#                         if e < s:
#                             loss = criterion(outputs, refs.to(device))
#                         else:
#                             loss = criterion(outputs[:, j, :], refs[:, j, :].to(device))
#                         loss.backward(retain_graph=True)
#                         #optimizer.step()
#                         #optimizer.zero_grad()

                
                
                
                
                
                
                
                
                
#                 if e < s: # s is the time we want to use all the traj to compute the loss, after s epochs the model evaluates the loss only on the last point
#                     validloss = criterion(outputs, refs.to(device))
#                 else:
#                     if (e==s):
#                         print(" Changing to evaluate loss only with the last point  ")
#                     validloss = criterion(outputs[-1,0,:], refs[-1,0,:].to(device))
#                 #print(" gave value to validloss")
#             valid_loss.append(validloss.item())
        
        
        
#         e += 1
#         if e % (nepochs // nprints) == 0:
#             print(e, "train loss", train_loss[-1], "valid loss",
#                 valid_loss[-1])
#         if save:
#             if validloss.item() < min_perf :
#                 min_perf = validloss.item()
#                 name='epochs=' +str(nepochs)+'_H='+str(history)+'_F='+str(future)+'_bsize='+str(bsize)+'_run_'+str(run)
#                 th.save(model.state_dict(), './'+str(folder)+'/models/'+name)
#                 bestmodel=model.load_state_dict(th.load('./'+str(folder)+'/models/'+name))
#     model=bestmodel
#     return train_loss, valid_loss

# ######################--------MODEL INIT----#################



class Sequence(nn.Module):
    def __init__(self,
                 hidden=50,
                 layer=1,
                 nfeatures=3,
                 dropout=0,
                 device=device):                #changed here the device 
        print("inside seq device=",device)
        super(Sequence, self).__init__()
        self.hidden = hidden
        self.layer = layer
        self.nfeatures = nfeatures
        self.device = device
        self.lstm1 = nn.LSTM(self.nfeatures,
                             self.hidden,
                             self.layer,
                             dropout=dropout).to(device)
        self.linear = nn.Linear(self.hidden, self.nfeatures).to(device)
        self.h_0 = nn.Parameter(th.randn(
            layer,
            1,
            hidden,
        ).to(device) / 100)
        self.c_0 = nn.Parameter(th.randn(
            layer,
            1,
            hidden,
        ).to(device) / 100)

    def forward(self, input, h_t=None, c_t=None):
        L, B, F = input.shape
        if h_t is None:
            h_t = self.h_0
        if c_t is None:
            c_t = self.c_0
        self.lstm1.flatten_parameters()
        out, (h_t, c_t) = self.lstm1(input, (h_t, c_t))
        output = out.view(input.size(0), input.size(1), self.hidden)
        output = th.sigmoid(self.linear(out)) #changed with sigmoid

        output = output.view(input.size(0), input.size(1), self.nfeatures).to(device)   #added to(device)    
        return output, (h_t, c_t)

    def get_init(self, bsize=1):
        assert bsize > 0
        if bsize == 1:
            return self.h_0, self.c_0
        return self.h_0.repeat(1, bsize, 1), self.c_0.repeat(1, bsize, 1)

    def generate(self, init_points, N, h_t=None, c_t=None, full_out=False):
        """
        - the init_points are used to initialize the memory of the LSTM
        - N is the number of generated points by the model on its own
        
        Return a tensor of size (N+L, 1, 12)
        with L the length of init_points
        """
        L, B, _ = init_points.shape
        if h_t is None or c_t is None:
            h_t, c_t = self.get_init(B)
        outl = N
        offset = 0
        if full_out:
            outl = N + L
            offset = L
        output = th.zeros(outl, B, self.nfeatures).to(device)
        # the init_points are used to initialize the memory of the LSTM
        init_pred, (h_t, c_t) = self.forward(init_points.to(device), h_t, c_t)
        if full_out:
            output[:offset] = init_pred
        #print("generate", h_t.shape,c_t.shape,init_pred[-1].shape)
        inp = init_pred[-1].unsqueeze(0).to(device)
        for i in range(offset, N + offset):
            inp, (h_t, c_t) = self.forward(inp.to(device), h_t, c_t)
            output[i] = inp
        return output, (h_t, c_t)



    def generate_cheap(self, init_points, N, h_t=None, c_t=None, full_out=False):
        """
        - the init_points are used to initialize the memory of the LSTM
        - N is the number of generated points by the model on its own
        
        Return a tensor of size (N+L, 1, 12)
        with L the length of init_points
        """
        with th.no_grad():
            L, B, _ = init_points.shape
            if h_t is None or c_t is None:
                h_t, c_t = self.get_init(B)
            outl = N
            offset = 0
            if full_out:
                outl = N + L
                offset = L
            output = th.zeros(outl, B, self.nfeatures).to(device)
            # the init_points are used to initialize the memory of the LSTM
            init_pred, (h_t, c_t) = self.forward(init_points.to(device), h_t, c_t)
            if full_out:
                output[:offset] = init_pred
            #print("generate", h_t.shape,c_t.shape,init_pred[-1].shape)
            inp = init_pred[-1].unsqueeze(0).to(device)
            for i in range(offset, N + offset):
                inp, (h_t, c_t) = self.forward(inp.to(device), h_t, c_t)
                output[i] = inp
                h_t.detach()
                c_t.detach()
        return output, (h_t, c_t)

    
    def generate_new(self, init_points, N, h_t=None, c_t=None, full_out=False,train=True):
        """
        - the init_points are used to initialize the memory of the LSTM
        - N is the number of generated points by the model on its own
        
        Return a tensor of size (N+L, 1, 12)
        with L the length of init_points
        """
        L, B, _ = init_points.shape
        if h_t is None or c_t is None:
            h_t, c_t = self.get_init(B)
        outl = N
        offset = 0
        if full_out:
            outl = N + L
            offset = L
        output = th.zeros(outl, B, self.nfeatures).to(device)
        # the init_points are used to initialize the memory of the LSTM
        init_pred, (h_t, c_t) = self.forward(init_points.to(device), h_t, c_t)
        if full_out:
            output[:offset] = init_pred
        #print("generate", h_t.shape,c_t.shape,init_pred[-1].shape)
        inp = init_pred[-1].unsqueeze(0).to(device)
        S=np.zeros([N+offset,L])
        for i in range(offset, N + offset):
            inp, (h_t, c_t) = self.forward(inp.to(device), h_t, c_t)
            output[i] = inp
            #-------------------TRY------------------#
            h_t.detach()
            c_t.detach()
            if (train==False):
                order=2
                if (i>5):
                    S[i]+=np.power(inp.cpu(),order)
        if (train==False):
            print(S/(N-5))
        return output, (h_t, c_t)    


class MLP(nn.Module):
    def __init__(self,
                 neurons=100,
                 nfeatures=12,
                 h=1,
                 f=25,
                 dropout=0,
                 device=device):                #changed here the device 
        print("inside seq device=",device)
        super(MLP, self).__init__()
        self.history=h
        self.future=f
        self.neurons = neurons
        self.nfeatures = nfeatures
        self.device = device

        self.MLP = nn.Sequential(
            nn.Linear(self.nfeatures*h, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, self.neurons+100),
            nn.ReLU(),
            nn.Linear(self.neurons+100,self.nfeatures*f),
            nn.Sigmoid()
        ).to(device)    

    """ for module in self.MLP.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)
    """
    def forward(self, input):

        output = self.MLP(input.to(device)).flatten()
        #print("forward output size =",output.view(-1, self.nfeatures *self.future).shape)
        return output.view(-1, self.nfeatures*self.future)

    def generate(self, init_points, N, h_t=None, c_t=None, full_out=False):
        L, B, _ = init_points.shape
        outl = N
        offset = 0
        if full_out:
            outl = N + L
            offset = L
        output = th.zeros(outl, B, self.nfeatures).to(device)
        if full_out:
            output[:offset] = init_points
        # inp is of expected size 1, B, D (number of features)
        inp = init_points[-1].unsqueeze(0).to(device)
        for i in range(offset, N + offset):
            inp = self.forward(inp.to(device))
            output[i] = inp
        return output

    
