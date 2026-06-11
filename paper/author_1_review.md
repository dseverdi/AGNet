I have read the paper and have few issues:


Critical issues:
1. The table :  held_out evaluation on dev_test is showing olniy cov \geq 0.95 but in the text further down the claim is that probl with encoder is better. It is not visible from the table to that extent. also SetPredictor full has meand +- std how come setPredicotr no-encoder doent
2. "The same singal is visilbe to a  probe ..." I dont  understand this part why it is important. 
3. We havent clearly mentioned our supervised training of PointerNetwork in supervised regime using solutions for AGPVG. Note that  solutions can be many and we ended up with worse results.
4. It seems to me that encoder + probe is what matters. There are papers that claim the inherent problem of PointerNet to generalize to larger instances (follow up papers after Vinalys convex hull example). Some reviewer could claim that why do you need Decoder in the first plade.
5. You are mentioning Wilson 95% and McNemar test,  I cannot find it clearly in the paper. 
6. Figures and tables are all over the place. This needs to be tied closely to the claims in the paper. Numbers should be visibile in the tables as proofs for claims. 
7.  I dont get the ROC-AUC etc how they showed up.
8. I also have publication that deals with terrain guarding with some variants. It would be usefull  to add it into the paper somehow. Here is the link: https://arxiv.org/abs/0809.0159

General issues:
1. Paper seems overall to verbose with lot of repetititons. Probably we'll have to truncate it for the jounral version. 
2. Dont overcomplicate with the statements let the flow be easy readable. 



Be very carefull how you will update this. Verify my points after you finish. 