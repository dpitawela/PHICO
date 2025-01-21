# PHICO: Personalised Human-AI Cooperative Classification Using Augmented Noisy Labels and Model Prediction

This repository contains the code for the paper --> link to add

PHICO basic model architecture


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


Loss function

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
