# PRIOR GUIDE: TEST-TIME PRIOR ADAPTATION FOR SIMULATION-BASED INFERENCE - Backmatter

---

## ACKNOWLEDGEMENTS

This work was a part of Finland's Ministry of Education and Culture's Doctoral Education Pilot under Decision No. VN/3137/2024- OKM- 6 (The Finnish Doctoral Program Network in Artificial Intelligence, AI- DOC). The project was also supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence, FCAI). NL was funded by Business Finland (project 3576/31/2023) and LUMI AI Factory (EU Horizon Europe Joint Undertaking and its members including top- up funding by Ministry of Education and Culture). LA was supported by Research Council of Finland grants 356498 and 358980. SR, MH, and AS acknowledge funding from the Research Council of Finland (grants 339730, 362408, 334600). The authors also acknowledge the research environment provided by ELLIS Institute Finland.

We acknowledge CSC - IT Center for Science, Finland, for computational resources provided by the LUMI supercomputer, owned by the EuroHPC Joint Undertaking and hosted by CSC and the LUMI consortium (LUMI projects 462000864 and 462000873). Access was provided through the Finnish LUMI- OKM allocation. We acknowledge the computational resources provided by the Aalto Science- IT project.

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the granting authority. Neither the European Union nor the granting authority can be held responsible for them.

> **Image description.** A horizontal image displaying two distinct sections, each representing a funding source or organization.
>
> The image is divided into two main parts:
>
> 1.  **Left Section (European Union Funding):**
>     *   This section features the flag of the European Union, which consists of a deep blue rectangular field containing a circle of twelve yellow stars.
>     *   Below the flag, the text "Co-funded by the European Union" is written in a dark, sans-serif font.
>
> 2.  **Right Section (EuroHPC Joint Undertaking):**
>     *   This section features a stylized logo representing a network or cluster. The graphic is composed of numerous thin, interconnected lines and nodes, forming a complex, circular, and somewhat radial structure.
>     *   To the right of this graphic, the text "EuroHPC Joint Undertaking" is displayed in a dark, sans-serif font.
>
> The two sections are positioned side-by-side, indicating a joint acknowledgment of funding and support from these entities. The overall color palette is dominated by the blue and yellow of the EU flag, contrasted with the dark text and the intricate lines of the EuroHPC logo.

## ETHICS STATEMENT

This work uses only synthetic datasets, with no sensitive data involved. The methods are for research purposes and pose no foreseeable ethical risks. We have followed the ICLR Code of Ethics.

## REPRODUCIBILITY STATEMENT

The code is available at https://github.com/acerbilab/prior- guide. All experiments use synthetic datasets. Algorithmic details are presented in Appendix A, and all experimental details are specified in Appendix C.

## REFERENCES

Luigi Acerbi. Variational Bayesian Monte Carlo. Advances in Neural Information Processing Systems (NeurIPS), 31, 2018.

Luigi Acerbi, Kalpana Dokka, Dora E Angelaki, and Wei Ji Ma. Bayesian comparison of explicit and implicit causal inference strategies in multisensory heading perception. PLoS computational biology, 14(7):e1006110, 2018.

Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic- dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In International Conference on Learning Representations.

Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic- DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In International Conference on Learning Representations (ICLR), 2022.

Benjamin Boys, Mark Girolami, Jakiv Pidstrigach, Sebastian Reich, Alan Mosca, and Omer Deniz Akyildiz. Tweedie moment projected diffusions for inverse problems. Transactions on Machine Learning Research.

Wessel P Bruinsma, Stratis Markou, James Requeima, Andrew YK Foong, Tom R Andersson, Anna Vaughan, Anthony Buonomo, J Scott Hosking, and Richard E Turner. Autoregressive conditional neural processes. In International Conference on Learning Representations (ICLR), 2023.

Gabriel Cardoso, Sylvain Le Corff, Eric Moulines, et al. Monte carlo guided denoising diffusion models for bayesian linear inverse problems. In The Twelfth International Conference on Learning Representations.

Paul E Chang, Nasrulloh Loka, Daolang Huang, Ulpu Remes, Samuel Kaski, and Luigi Acerbi. Amortized probabilistic conditioning for optimization, simulation and inference. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR, 2025.

Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In International Conference on Learning Representations (ICLR), 2023.

Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation- based inference. Proceedings of the National Academy of Sciences, 117(48):30055- 30062, 2020.

Marco Del Negro and Frank Schorfheide. Forming priors for DSGE models (and how it affects the assessment of nominal rigidities). Journal of Monetary Economics, 55(7):1191- 1208, 2008.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pp. 8780- 8794. Curran Associates, Inc., 2021.

Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. In The Twelfth International Conference on Learning Representations, 2024.

Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl- Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy- based diffusion models and mcmc. In International conference on machine learning, pp. 8489- 8510. PMLR, 2023.

Lasse Elsemüller, Hans Olischläger, Marvin Schmitt, Paul- Christian Bürkner, Ullrich Köthe, and Stefan T Radev. Sensitivity- aware amortized Bayesian inference. Transactions on Machine Learning Research (TMLR), 2024.

Marc Anton Finzi, Anudhyan Boral, Andrew Gordon Wilson, Fei Sha, and Leonardo Zepeda- Núñez. User- defined event sampling and uncertainty quantification in diffusion models for physical dynamical systems. In Proceedings of the International Conference on Machine Learning (ICML), pp. 10136- 10152. PMLR, 2023.

Seth Flaxman, Swapnil Mishra, Axel Gandy, H Juliette T Unwin, Thomas A Mellan, Helen Coupland, Charles Whittaker, Harrison Zhu, Tresnia Berah, Jeffrey W Eaton, et al. Estimating the effects of non- pharmaceutical interventions on covid- 19 in europe. Nature, 584(7820):257- 261, 2020.

Daniel Foreman- Mackey. corner.py: Scatterplot matrices in python. The Journal of Open Source Software, 1(2):24, jun 2016. doi: 10.21105/joss.00024. URL https://doi.org/10.21105/joss.00024.

Marta Garnelo, Dan Rosenbaum, Chris J Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J Rezende, and SM Ali Eslami. Conditional neural processes. In Proceedings of the International Conference on Machine Learning (ICML), pp. 1704- 1713, 2018.

Tomas Geffner, George Papamakarios, and Andriy Mnih. Compositional score modeling for simulation- based inference. In International Conference on Machine Learning, pp. 11098- 11116. PMLR, 2023.

Andrew Gelman, John B Carlin, Hal S Stern, Aki Vehtari, and Donald B Rubin. Bayesian data analysis, volume 3rd edition. Chapman and Hall/CRC, 2013.

Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul- Christian Bürkner, and Martin Modrák. Bayesian workflow. arXiv preprint arXiv:2011.01808, 2020.

Manuel Gloeckler, Michael Deistler, Christian Weilbach, Frank Wood, and Jakob H Macke. All- in- one simulation- based inference. In Proceedings of the International Conference on Machine Learning (ICML), pp. 15735- 15766, 2024.

David Greenberg, Marcel Nonnenmacher, and Jakob Macke. Automatic posterior transformation for likelihood- free inference. In Proceedings of the International Conference on Machine Learning (ICML), pp. 2404- 2414. PMLR, 2019.

Joeri Hermans, Volodimir Begy, and Gilles Louppe. Likelihood- free mcmc with amortized approximate ratio estimators. In International conference on machine learning, pp. 4239- 4248. PMLR, 2020.

Jonathan Ho and Tim Salimans. Classifier- free diffusion guidance. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications., 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 6840- 6851. Curran Associates, Inc., 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 18954- 18967. Curran Associates, Inc., 2022.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319- 326, 2025.

Daolang Huang, Ayush Bharti, Amauri Souza, Luigi Acerbi, and Samuel Kaski. Learning robust statistics for simulation- based inference under model misspecification. In Advances in Neural Information Processing Systems (NeurIPS), volume 36. Curran Associates, Inc., 2024.

Bobby Huggins, Chengkun Li, Marlon Tobaben, Mikko J Aarnos, and Luigi Acerbi. PyVBMC: Efficient Bayesian inference in Python. Journal of Open Source Software, 8(86):5428, 2023.

Aapo Hyvärinen and Peter Dayan. Estimation of non- normalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion- based generative models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 26565- 26577. Curran Associates, Inc., 2022.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

Konrad P Körding, Ulrik Beierholm, Wei Ji Ma, Steven Quartz, Joshua B Tenenbaum, and Ladan Shams. Causal inference in multisensory perception. PLoS One, 2(9):e943, 2007.

Cheuk Kit Lee, Paul Jeha, Jes Frellsen, Pietro Lio, Michael Samuel Albergo, and Francisco Vargas. Debiasing guidance for discrete diffusion with sequential monte carlo. arXiv preprint arXiv:2502.06079, 2025.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out- of- distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems (NeurIPS), volume 31. Curran Associates, Inc., 2018.

Julia Linhart, Gabriel Victorino Cardoso, Alexandre Gramfort, Sylvain Le Corff, and Pedro LC Rodrigues. Diffusion posterior sampling for simulation- based inference in tall data settings. arXiv preprint arXiv:2404.07593, 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben- Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Lorenzo Loconte, Aleksanteri M. Sladek, Stefan Mengel, Martin Trapp, Arno Solin, Nicolas Gillis, and Antonio Vergari. Subtractive mixture models via squaring: Representation and learning. In International Conference on Learning Representations (ICLR), 2024. David Lopez- Paz and Maxime Oquab. Revisiting classifier two- sample tests. In International Conference on Learning Representations (ICLR), 2017. Jan- Matthias Lueckmann, Pedro J Goncalves, Giacomo Bassetto, Kaan Öcal, Marcel Nonnenmacher, and Jakob H Macke. Flexible statistical inference for mechanistic models of neural dynamics. In Advances in Neural Information Processing Systems (NeurIPS), volume 30. Curran Associates, Inc., 2017. Jan- Matthias Lueckmann, Giacomo Bassetto, Theofanis Karaletsos, and Jakob H Macke. Likelihood- free inference with emulator networks. In Symposium on Advances in Approximate Bayesian Inference, pp. 32- 53. PMLR, 2019. Jan- Matthias Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke. Benchmarking simulation- based inference. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS), Proceedings of Machine Learning Research, pp. 343- 351. PMLR, 2021. Hila Manor and Tomer Michaeli. On the posterior distribution in denoising: Application to uncertainty quantification. In The Twelfth International Conference on Learning Representations. Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, and Marcus Brubaker. Amortized in- context Bayesian posterior estimation. arXiv preprint arXiv:2502.06601, 2025. Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations (ICLR), 2022. Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don't know? In International Conference on Learning Representations (ICLR), 2019. Tung Nguyen and Aditya Grover. Transformer Neural Processes: Uncertainty- aware meta learning via sequence modeling. In Proceedings of the International Conference on Machine Learning (ICML), pp. 123- 134. PMLR, 2022. George Papamakarios and Iain Murray. Fast \(\epsilon\) - free inference of simulation models with Bayesian conditional density estimation. Advances in Neural Information Processing Systems (NeurIPS), 29, 2016. George Papamakarios, David Sterratt, and Iain Murray. Sequential neural likelihood: Fast likelihood- free inference with autoregressive flows. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), Proceedings of Machine Learning Research, pp. 837- 848. PMLR, 2019. Troels Pedersen. Stochastic multipath model for the in- room radio channel based on room electromagnetics. IEEE Transactions on Antennas and Propagation, 67(4):2591- 2603, 2019. Xinyu Peng, Ziyang Zheng, Wenrui Dai, Nuoqian Xiao, Chenglin Li, Junni Zou, and Hongkai Xiong. Improving diffusion models for inverse problems using optimal posterior covariance. In International Conference on Learning Representations (ICLR), 2024. Stefan T Radev, Ulf K Mertens, Andreas Voss, Lynton Ardizzone, and Ullrich Kothe. Bayesflow: Learning complex stochastic models with invertible neural networks. IEEE Transactions on Neural Networks and Learning Systems, 33(4):1452- 1466, 2020. Severi Rissanen, Markus Heinonen, and Arno Solin. Free hunch: Denoiser covariance estimation for diffusion models without extra costs. In International Conference on Learning Representations (ICLR), 2025.

Christian P Robert. The Bayesian Choice: From Decision- theoretic Foundations to Computational Implementation, volume 2nd edition. Springer, 2007. Christian P. Robert and George Casella. Monte Carlo Statistical Methods. Springer Texts in Statistics. Springer, New York, 2nd edition, 2004. ISBN 0- 387- 21239- 6. François Rozet, Gérôme Andry, François Lanusse, and Gilles Louppe. Learning diffusion priors from observations by expectation maximization. Advances in Neural Information Processing Systems, 37:87647- 87682, 2024. Simo Särkkä and Arno Solin. Applied Stochastic Differential Equations. Cambridge University Press, 2019. Marvin Schmitt, Paul- Christian Bürkner, Ullrich Köthe, and Stefan T Radev. Detecting model misspecification in amortized Bayesian inference with neural networks. In DAGM German Conference on Pattern Recognition, pp. 541- 557. Springer, 2023. Marvin Schmitt, Valentin Prat, Ullrich Köthe, Paul- Christian Bürkner, and Stefan T. Radev. Consistency models for scalable and fast simulation- based inference. In Advances in Neural Information Processing Systems (NeurIPS), 2024. Louis Sharrock, Jack Simons, Song Liu, and Mark Beaumont. Sequential neural score estimation: Likelihood- free inference with conditional score based diffusion models. In International Conference on Machine Learning, pp. 44565- 44602. PMLR, 2024. Steven C Sherwood, Mark J Webb, James D Annan, Kyle C Armour, Piers M Forster, Julia C Hargreaves, Gabriele Hegerl, Stephen A Klein, Kate D Marvel, Eelco J Rohling, et al. An assessment of earth's climate sensitivity using multiple lines of evidence. Reviews of Geophysics, 58(4):e2019RG000678, 2020. Francesco Silvestrin, Chengkun Li, and Luigi Acerbi. Stacking variational Bayesian Monte Carlo. arXiv preprint arXiv:2504.05004, 2025. Marta Skreta, Tara Akhound- Sadegh, Viktor Ohanesian, Roberto Bondesan, Alan Aspuru- Guzik, Arnaud Doucet, Rob Brekelmans, Alexander Tong, and Kirill Neklyudov. Feynman- kac correctors in diffusion: Annealing, guidance, and product of experts. In Forty- second International Conference on Machine Learning. Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test- time compute optimally can be more effective than scaling model parameters. In International Conference on Learning Representations (ICLR), 2025. Jascha Sohl- Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the International Conference on Machine Learning (ICML), pp. 2256- 2265. PMLR, 2015. Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse- guided diffusion models for inverse problems. In International Conference on Learning Representations (ICLR), 2023a. Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming- Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss- guided diffusion models for plug- and- play controllable generation. In Proceedings of the International Conference on Machine Learning (ICML), pp. 32483- 32498. PMLR, 2023b. Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. Yang Song, Jascha Sohl- Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score- based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR). ICLR, May 2021. H.W. Sorenson and D.L. Alspach. Recursive Bayesian estimation using Gaussian sums. Automatica, 7(4):465- 479, 1971. ISSN 0005- 1098.

Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. Validating Bayesian inference algorithms with simulation- based calibration. arXiv preprint arXiv:1804.06788, 2018.

Owen Thomas, Ritabrata Dutta, Jukka Corander, Samuel Kaski, and Michael U Gutmann. Likelihood- free inference by ratio estimation. Bayesian Analysis, 17(1):1- 31, 2022.

James Thornton, Louis Béthune, Ruixiang ZHANG, Arwen Bradley, Preetum Nakkiran, and Shuangfei Zhai. Composition and control with distilled energy diffusion models and sequential monte carlo. In The 28th International Conference on Artificial Intelligence and Statistics.

George L Turin, Fred D Clapp, Tom L Johnston, Stephen B Fine, and Dan Lavry. A statistical model of urban multipath propagation. IEEE Transactions on Vehicular Technology, 21(1):1- 9, 1972.

George E Uhlenbeck and Leonard S Ornstein. On the theory of the brownian motion. Physical Review, 36(5):823, 1930.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30. Curran Associates, Inc., 2017.

Aki Vehtari, Daniel Simpson, Andrew Gelman, Yuling Yao, and Jonah Gabry. Pareto smoothed importance sampling. Journal of Machine Learning Research, 25(72):1- 58, 2024.

Julius Vetter, Manuel Gloeckler, Daniel Gedon, and Jakob H Macke. Effortless, simulation- efficient Bayesian inference using tabular foundation models. arXiv preprint arXiv:2504.17660, 2025.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661- 1674, 2011.

George Whittle, Juliusz Ziomek, Jacob Rawling, and Michael A Osborne. Distribution transformers: Fast approximate Bayesian inference with on- the- fly prior adaptation. arXiv preprint arXiv:2502.02463, 2025.

Frank Wilcoxon. Individual comparisons by ranking methods. Biometrics Bulletin, 1(6):80- 83, 1945.

Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen Green, Jakob H Macke, and Bernhard Schölkopf. Flow matching for scalable simulation- based inference. In Advances in Neural Information Processing Systems (NeurIPS), volume 36. Curran Associates, Inc., 2024.

Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. Advances in Neural Information Processing Systems, 36:31372- 31403, 2023.

Wang Yuyan, Michael Evans, and David J Nott. Robust Bayesian methods using amortized simulation- based inference. arXiv preprint arXiv:2504.09475, 2025.

---

*Transcribed with OCR and VLMs; text, equations, and figure descriptions may contain mistakes.*
