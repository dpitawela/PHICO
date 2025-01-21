# PHICO: Personalised Human-AI Cooperative Classification Using Augmented Noisy Labels and Model Prediction

This repository contains the code for the paper --> link to add

* Below are basic code snippets to provide the idea of model, training loop, loss function and data loader.
* A complete code taking CIFAR10N experiment as example is in the 'Codes' folder that goes though python notebook 1 to 5.
* code/matrics.py has the implementations of the evaluation criteria.

### Basic model architecture

	class  AdaptedAI(nn.Module):
		def  __init__(self):
			super(AdaptedAI, self).__init__()
			
			# base model
			# can be any model architecture of choice that supports classification
			# the output should be the classification head logits
			self.base_model = 
			
			# human label encoder
			self.n_l_encoder = nn.Sequential(
			nn.Linear(10, 32),
			nn.ReLU(),
			nn.Linear(32, 10)
			)  
			
			# decision model
			self.decision_ai = nn.Sequential(
			nn.Linear(20, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 10),
			)
		
		def  forward(self, imgs, n_l):
			img_features = self.base_model(imgs)
			n_l_features = self.n_l_encoder(n_l)
			out = torch.cat((img_features, n_l_features), dim=1)
			out = self.decision_ai(out)
			return img_features, n_l, out

### Basic data loader

    class  CIFAR10(VisionDataset):
	    def  __init__(self):
		    # data loading and etc
	    
	    def  __len__(self) -> int:
		    return  len(self.noisyLabels)
	    
	    def  __getitem__(self, index: int) -> Tuple[Any, Any]:
		    # data loading covers all the noisy labels from the cluster.
		    
		    # image and consensus label loading is adjusted accordingly.
		    img, consensus, n_l = self.data[index % len(self.data)], self.consensus[index % len(self.data)], self.noisyLabels[index]
		    img = Image.fromarray(img)
		    
		    #should return (image, consensus, noisy label)
		    return img, target, n_l
			
### Loss function

    class  CorrectionLoss(nn.Module):
    	def  __init__(self, loss1, C=0, N_human=None, N_base=None):
    		super().__init__()
    		self.loss1 = loss1
    		self.C = C
    		self.N_h = N_human
    		self.N_b = N_base

		def  noiseCorrection(self, prediction, n_input):
			softmax_pred = F.softmax(prediction, dim=1)
			correction = torch.tensor([]).to(device=device)

			if  self.N_h is  not  None:
				loss_h = F.nll_loss(torch.log(torch.matmul(softmax_pred, self.N_h)), n_input, reduction='mean')
				correction = torch.cat((correction, torch.tensor([loss_h]).to(device)))

			if  self.N_b is  not  None:
				loss_b = F.nll_loss(torch.log(torch.matmul(softmax_pred, self.N_b)), n_input, reduction='mean')
				correction = torch.cat((correction, torch.tensor([loss_b]).to(device)))

			return  self.C * torch.mean(correction)

		def  forward(self, prediction, n_input, target):
			l = self.loss1(prediction, target)
			correction = self.noiseCorrection(prediction, n_input)
			return l+correction

### Basic training loop

    criterion = CorrectionLoss(nn.CrossEntropyLoss(), 0.1, noise_H)
    
    EPOCHS = 100
        for _ in  range(EPOCHS):
	        for i, (imgs, labels, n_labels) in  enumerate(trainloader):
		        imgs = imgs.to(device)
		        labels = labels.to(device)
		        n_labels = n_labels.to(device)
		        n_labels = F.one_hot(n_labels, num_classes=10).to(device, dtype=torch.float32)

		        optimizer.zero_grad()
		        _, _, outputs = adapt_model(imgs, n_labels)

		        loss = criterion(outputs, torch.argmax(n_labels, dim=1), labels)
		        loss.backward()
		        optimizer.step()

		        _, preds = torch.max(outputs, 1)
		        global_step += 1