# MuonSSM Vision Spartial Modeling

You just need to change the self.mixer by other MuonSSM model in mamba_vision.py file, and can reproduce all the experiment results follow by the MambaVision-Tiny version

```bash
self.mixer = MambaVisionMixer(d_model=dim, 
                                d_state=8,  
                                d_conv=3,    
                                expand=1
                                )
# self.mixer = MuonMambaVisionMixer(d_model=dim, 
#                               d_state=8,  
#                               d_conv=3,    
#                               expand=1
#                               )
# self.mixer = MuonLonghornVisionMixer(d_model=dim, 
#                               d_state=8,  
#                               d_conv=3,    
#                               expand=1
#                               )
```