https://docs.google.com/document/d/1-nHWaiJVDi3Bl4YhC5iJRgnldpQVw9B3/edit


@ 18 / 20 / 2023
Hi Gustavo,
This is just an update about what me and Tim talked in the last meeting and seeking what you think about this side.
During the last meeting,  we had a chat about an idea related to “Ordinal Classification”, inspired by these papers,
Rank-N-Contrast: Learning Continuous Representations for Regression - https://arxiv.org/abs/2210.01189
Improving Deep Regression with Ordinal Entropy - https://arxiv.org/abs/2301.08915
Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation - https://openreview.net/pdf?id=ij3svnPLzG
 
Most of the times ML techniques are applied to medical classifications such as cancer grading, Gleason score, PI-RADS score as usual classifications tasks forgetting that the grade itself has a meaning/order. We may be able to propose a solution for this.
 
Found some classification datasets that has ordinal relationships,
Knee Osteoarthritis Severity Grading Dataset - https://data.mendeley.com/datasets/56rmx5bjcr/1/
Prostate cANcer graDe Assessment - https://panda.grand-challenge.org/home/
Diabetic Retinopathy Images Database - https://academictorrents.com/details/99811ba62918f8e73791d21be29dcc372d660305
Mammography - https://vindr.ai/datasets/mammo
 
To make the problem a bit more interesting, if there are noisy datasets that has ordinal relationships, we may be able to try “Ordinal Classification with Noisy Labels”. I’m not sure if the above datasets contain noisy labels though.
 
How do you feel about this idea/area..? and have you seen benchmark datasets related to this area, that I can use to try something.?
