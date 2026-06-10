# PRIOR GUIDE: TEST-TIME PRIOR ADAPTATION FOR SIMULATION-BASED INFERENCE

Yang Yang \(^{1}\) , Severi Rissanen \(^{2,3}\) , Paul E. Chang \(^{1,4}\) , Nasrulloh Loka \(^{1}\) , Daolang Huang \(^{2,3}\) , Arno Solin \(^{2,3}\) , Markus Heinonen \(^{3}\) , Luigi Acerbi \(^{1}\)

\(^{1}\) Department of Computer Science, University of Helsinki, Finland \(^{2}\) ELLIS Institute Finland \(^{3}\) Department of Computer Science, Aalto University, Finland \(^{4}\) DataCrunch {yang.yang,paul.chang,nasrulloh.satriol,luigi.acerbi}@helsinki.fi {severi.rissanen,daolang.huang,arno.solin,markus.o.heinonen}@aalto.fi

## ABSTRACT

Amortized simulator- based inference offers a powerful framework for tackling Bayesian inference in computational fields such as engineering or neuroscience, increasingly leveraging modern generative methods like diffusion models to map observed data to model parameters or future predictions. These approaches yield posterior or posterior- predictive samples for new datasets without requiring further simulator calls after training on simulated parameter- data pairs. However, their applicability is often limited by the prior distribution(s) used to generate model parameters during this training phase. To overcome this constraint, we introduce PriorGuide, a technique specifically designed for diffusion- based amortized inference methods. PriorGuide leverages a novel guidance approximation that enables flexible adaptation of the trained diffusion model to new priors at test time, crucially without costly retraining. This allows users to readily incorporate updated information or expert knowledge post- training, enhancing the versatility of pre- trained inference models.

## 1 INTRODUCTION

Simulation- based inference (SBI) has become a key tool across scientific disciplines, enabling Bayesian inference for complex systems where the likelihood function \(p(\mathbf{x} | \boldsymbol {\theta})\) for data \(\mathbf{x}\) and model parameters \(\boldsymbol{\theta}\) is intractable, but one can easily simulate from the forward model \(\mathbf{x} \sim p(\mathbf{x} | \boldsymbol {\theta})\) (Cranmer et al., 2020). Within the Bayesian paradigm, prior beliefs about parameters \(\boldsymbol{\theta}\) are updated with observed data \(\mathbf{x}\) to form a posterior distribution \(p(\boldsymbol {\theta} | \mathbf{x})\) (Gelman et al., 2013). While traditional methods like Markov Chain Monte Carlo (MCMC; Robert, 2007) are effective when likelihoods are available, recent amortized inference methods can directly learn the inverse mapping \(\mathbf{x} \mapsto p(\boldsymbol {\theta} | \mathbf{x})\) from observations to posteriors using neural networks (Lueckmann et al., 2017; Radev et al., 2020). These amortized approaches, once trained, can rapidly infer posterior (parameters) or posterior- predictive (data) distributions given new observations, significantly speeding up the inference process.

Modern generative models, including diffusion models (Sohl- Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021), transformers (Vaswani et al., 2017), and flow- matching techniques (Lipman et al., 2023), have demonstrated state- of- the- art performance in tackling these inverse modeling tasks for amortized SBI (Müller et al., 2022; Wildberger et al., 2024; Schmitt et al., 2024; Gloeckler et al., 2024; Chang et al., 2025; Whittle et al., 2025; Mittal et al., 2025; Hollmann et al., 2025). These models are trained on vast numbers of simulated parameter- data pairs \((\boldsymbol {\theta}, \mathbf{x})\) , often drawing parameters \(\boldsymbol{\theta}\) from a broad, uniform prior distribution to ensure comprehensive coverage of the parameter space.

However, this reliance on a fixed training prior introduces significant limitations. Practitioners often possess domain knowledge that, if incorporated as a more specific prior, could substantially improve inference accuracy. Moreover, prior sensitivity analysis—a crucial step for assessing the robustness of scientific conclusions to modeling assumptions—becomes cumbersome. This practice is vital across many disciplines, e.g., economists must validate the policy implications of macroeconomic models against different theoretical priors (Del Negro & Schorfheide, 2008), climate scientists need to validate the climate sensitivity estimates over multiple sets of assumptions (Sherwood et al., 2020), and epidemiologists need to assess the sensitivity of pandemic forecasts to assumptions

> **Image description.** A complex scientific figure, Figure 1, consisting of three vertical panels (columns) and two horizontal rows (plots), illustrating the process of adapting a diffusion model to new prior information for time-series inference.
>
> The figure is organized into three main columns, each representing a stage of the inference process: "Priors" (left), "True Posteriors" (middle), and "Sampled Posteriors" (right). Each column contains a top plot showing the parameter space ($\theta_1, \theta_2$) and a bottom plot showing the data space (Process value $x$ vs. Time step).
>
> **Panel 1: Priors (Left)**
> *   **Top Plot (Parameters):** This plot displays two concentric, slightly irregular ellipses representing prior distributions. The outer, larger distribution is colored orange and labeled "Training prior." The inner, smaller distribution is colored blue and labeled "New prior." A single black dot is positioned near the center, labeled "True parameters." The axes are labeled $\theta_1$ (horizontal) and $\theta_2$ (vertical).
> *   **Bottom Plot (Data):** This plot shows a single black line graph representing the "True time-series data." The process value $x$ starts high at the beginning of the time step and decays rapidly, leveling off toward the end.
>
> **Panel 2: True Posteriors (Middle)**
> *   **Top Plot (Parameters):** Similar to the first panel, this plot shows two concentric distributions: the outer orange "Training prior" and the inner blue "New prior." A single black dot marks the "True parameters." The axes are labeled $\theta_1$ and $\theta_2$.
> *   **Bottom Plot (Data):** This plot displays two overlapping line graphs representing posterior predictive distributions. The orange line, labeled "Posterior predictive (old)," is broader and higher than the blue line, labeled "Posterior predictive (new)." A single black line, representing the observed data, is visible, closely following the trend of the blue line.
>
> **Panel 3: Sampled Posteriors (Right)**
> *   **Top Plot (Parameters):** This plot shows a dense cloud of scattered points (samples) representing the posterior distributions. The samples are categorized into three groups: "PriorGuide low" (purple/pink), "PriorGuide high" (yellow/green), and "Diffusion model" (light blue/cyan). The cloud is centered around the region defined by the inner blue prior in the middle panel. The axes are labeled $\theta_1$ and $\theta_2$.
> *   **Bottom Plot (Data):** This plot shows three overlapping line graphs, corresponding to the three sample groups in the top plot. The lines for "PriorGuide low," "PriorGuide high," and "Diffusion model" all closely track the single black line representing the observed data.
>
> **Text and Labels:**
> The figure is titled "Figure 1: PriorGuide adapts a diffusion model to new prior information at test time for simulator-based inference—here for a time-series model." All axes are clearly labeled with their respective variables ($\theta_1, \theta_2, x$) and the common label "Time step" for the horizontal axis of the bottom plots. The labels within the plots accurately identify the different distributions and data points used in the simulation.

<center>Figure 1: PriorGuide adapts a diffusion model to new prior information at test time for simulator-based inference—here for a time-series model. Left: Original broad training prior and a new, more specific target prior. Middle: Corresponding true posterior distributions over parameters \((\uparrow)\) and predictive data \((\downarrow)\) for each prior, given observed data. Right: Posterior \((\uparrow)\) and posterior-predictive \((\downarrow)\) samples from the diffusion model trained on the old prior vs. those from PriorGuide at low or high test-time cost, illustrating PriorGuide's ability to match the new posterior from the middle column. </center>

about disease transmission (Flaxman et al., 2020). Changing priors is computationally challenging: non- amortized methods require costly re- simulation for each new prior, while amortized methods require retraining (Elsemüller et al., 2024). Though practitioners often resort to approximations like importance sampling to avoid these costs, such methods fail when priors differ substantially. This limitation becomes increasingly problematic as the field trends towards foundation models for SBI (Hollmann et al., 2025; Vetter et al., 2025). While some recent amortized methods offer inference- time prior adaptation, they are often restricted to specific families of priors (e.g., factorized histograms or Gaussian mixtures pre- defined at training) or simple constraints (Elsemüller et al., 2024; Chang et al., 2025; Whittle et al., 2025; Gloeckler et al., 2024). The broader issue is the impracticality of pre- training over all potential tasks, such as all prior distributions a user might wish to employ. Recent successes of the test- time compute paradigm (Snell et al., 2025) suggest that rather than attempting exhaustive amortization for all scenarios, models could be designed to flexibly incorporate specific requirements, such as a user- defined prior, through dedicated computations at inference time—an ability which was unattained so far. For a comprehensive discussion of the related work, we refer the reader to Appendix A.1.

In this paper, we introduce PriorGuide, a novel method that empowers diffusion- based amortized inference models with the ability to adapt to new prior distributions \(q(\theta)\) at inference time, without the need for retraining the original score model trained under prior \(p_{\mathrm{train}}(\theta)\) .<sup>1</sup> PriorGuide leverages the guidance mechanisms inherent in diffusion models to steer the generative process according to the new target prior, seamlessly integrating new user- provided information during the sampling process. Crucially, this approach allows for a trade- off: users can invest more computational resources at inference time—such as more diffusion steps or adding refinement techniques like Langevin dynamics—to achieve higher inference fidelity. See Fig. 1 for a conceptual overview.

Contributions Our main contribution is a principled framework for flexibly incorporating new prior distributions at inference time into pre- trained diffusion models for SBI. We leverage a novel Gaussian mixture model approximation for effectively turning the target prior \(q(\theta)\) into a tractable guidance term for the diffusion sampling process. We demonstrate empirically the effectiveness of PriorGuide on a range of SBI problems, showing its ability to accurately recover posterior and posterior- predictive distributions under various inference- time prior specifications. We show how sampling can be refined with additional Langevin dynamics steps, and study the tradeoff between test- time compute and inference accuracy within our framework.

## 2 BACKGROUND

The primary goal in many scientific applications is to infer model parameters \(\theta\) given observed data \(\mathbf{x}\) , or to predict future data \(\mathbf{x}^{*}\) . Bayesian inference provides a framework for computing the posterior distribution over parameters or posterior predictive distribution for new data:

\[\begin{array}{r l} & {p(\pmb {\theta}\mid \mathbf{x})\propto p(\mathbf{x}\mid \pmb {\theta})p(\pmb {\theta}),\qquad \mathrm{(posterior)}}\\ & {p(\mathbf{x}^{*}\mid \mathbf{x}) = \int p(\mathbf{x}^{*}\mid \pmb {\theta},\mathbf{x})p(\pmb {\theta}\mid \mathbf{x})\mathrm{d}\pmb {\theta},\quad \mathrm{(posterior~predictive)}} \end{array} \quad (1)\]

where \(p(\pmb {\theta})\) is the prior and \(p(\mathbf{x}\mid \pmb {\theta})\) is the likelihood.

The prior \(p(\pmb {\theta})\) encodes the practitioner's beliefs about plausible parameter values before observing data—beliefs informed by physical constraints, previous experiments, or theoretical considerations (Gelman et al., 2013). In practice, priors are typically simple—Gaussians, uniforms, smooth distributions—reflecting broad domain knowledge rather than precise specifications. Yet even among simple priors, the specific choice varies by application: different analyses, constraints, or domain knowledge call for different specifications, and modern statistical practice recommends validation of results under multiple prior choices (Gelman et al., 2020).

The likelihood \(p(\mathbf{x}\mid \pmb {\theta})\) encodes the statistical or mechanistic description of how data are generated from the model. In many real- world scenarios from finance to physics, evaluating the likelihood is intractable, but generating samples \(\mathbf{x}\sim p(\mathbf{x}\mid \pmb {\theta})\) from a simulator is feasible, leading to the field of Simulation- Based Inference (SBI; Cranmer et al., 2020).

A powerful paradigm within SBI is amortized inference. Instead of performing inference from scratch for each \(\mathbf{x}\) , amortized methods train a neural network \(q_{\phi}\) once on a large dataset of simulated parameter- data pairs, \(\mathcal{D}_{\mathrm{sim}} = \{(\pmb {\theta}_{i},\mathbf{x}_{i})\}_{i = 1}^{N}\) . Parameters \(\pmb{\theta}_{i}\) are drawn from a training prior, \(p_{\mathrm{train}}(\pmb {\theta})\) , and data \(\mathbf{x}_{i}\sim p(\mathbf{x}\mid \pmb {\theta}_{i})\) . Once trained, \(q_{\phi}\) provides rapid inference for new observations, amortizing the upfront computational cost. The network \(q_{\phi}\) is usually trained to approximate the posterior \(p(\pmb {\theta}\mid \mathbf{x})\) (Lueckmann et al., 2017; Greenberg et al., 2019; Radev et al., 2020), the likelihood \(p(\mathbf{x}\mid \pmb {\theta})\) (Papamakarios et al., 2019), or the posterior predictive distribution \(p(\mathbf{x}^{*}\mid \mathbf{x})\) (Garnelo et al., 2018; Müller et al., 2022). Recently proposed architectures can flexibly perform all of these tasks, using transformers (Chang et al., 2025) or diffusion models (Gloeckler et al., 2024).

### 2.1 DIFFUSION MODELS

Diffusion models are a powerful framework for generative modeling that transforms samples from arbitrary to simple distributions and vice versa through a gradual noising and denoising process (Sohl- Dickstein et al., 2015). In the forward diffusion process, starting from a distribution \(p(\mathbf{z}_{0})\) we can draw samples from (e.g., joint samples from the training prior and simulator), Gaussian noise is progressively added to the samples until, at the end of the process ( \(t = 1\) ), the distribution converges to a simple terminal distribution, typically Gaussian. In the Variance Exploding (VE) formulation (Song et al., 2021; Karras et al., 2022), the forward diffusion process can be described as:

\[p(\mathbf{z}_{t}) = \int p(\mathbf{z}_{t}\mid \mathbf{z}_{0})p(\mathbf{z}_{0})\mathrm{d}\mathbf{z}_{0} = \int \mathcal{N}(\mathbf{z}_{t}\mid \mathbf{z}_{0},\sigma (t)^{2}\mathbf{I})p(\mathbf{z}_{0})\mathrm{d}\mathbf{z}_{0}, \quad (2)\]

where \(p(\mathbf{z}_{t}\mid \mathbf{z}_{0})\) is the transition kernel (here Gaussian), \(\sigma (t)^{2}\) defines the noise variance schedule as a function of time (typically increasing with \(t\) ), and \(\mathbf{z}_{t}\) represents the noisy samples at time \(t\) . The corresponding reverse process reconstructs the original sample distribution from noise, and can be formulated as either a stochastic differential equation (SDE) or an ordinary differential equation (ODE). The reverse SDE process takes the form (Song et al., 2021; Karras et al., 2022):

\[\mathrm{d}\mathbf{z}_{t} = -2\hat{\sigma} (t)\sigma (t)\nabla_{\mathbf{z}}\log p(\mathbf{z}_{t})\mathrm{d}t + \sqrt{2\hat{\sigma} (t)\sigma(t)}\mathrm{d}\omega_{t}, \quad (3)\]

where \(\nabla_{\mathbf{z}}\log p(\mathbf{z}_{t})\) is the score function (gradient of the log- density), \(\mathrm{d}\omega_{t}\) is a Wiener process representing Brownian motion (noise), and \(\hat{\sigma} (t)\) is the time derivative of the noise schedule.

Learning the score function The score function \(\nabla_{\mathbf{z}}\log p(\mathbf{z}_{t})\) can be approximated using a neural network \(s(\mathbf{z}_{t},t)\) , trained to minimize the denoising score matching loss (Hyvärinen & Dayan, 2005; Vincent, 2011; Song et al., 2021):

\[\mathcal{L}_{\mathrm{DSM}} = \mathbb{E}_{t\sim p(t)}\mathbb{E}_{\mathbf{z}_{0}\sim p(\mathbf{z}_{0})}\mathbb{E}_{\mathbf{z}_{t}\sim \mathcal{N}(\mathbf{z}_{t}\mid \mathbf{z}_{0},\sigma (t)^{2}\mathbf{I})}\left[\omega (t)\left\| s(\mathbf{z}_{t},t) - \nabla_{\mathbf{z}_{t}}\log p(\mathbf{z}_{t}\mid \mathbf{z}_{0})\right\|_{2}^{2}\right]. \quad (4)\]

Here \(p(t)\) is the distribution of noise levels sampled during training and \(\omega (t)\) weights different noise levels in the loss. Once trained, the network \(s(\mathbf{z}_t, t)\) approximates the gradient of the log- probability density of noised distributions and affords sampling through the reverse SDE; Eq. (3). Starting from a sample \(\mathbf{z}_t \sim \mathcal{N}(\mathbf{z}_t | \mathbf{z}_0, \sigma_{\max}^2 \mathbf{I})\) for \(t = 1\) with sufficiently large \(\sigma_{\max}\) , integrating the reverse process backward in time approximately reconstructs the original distribution \(p(\mathbf{z}_0)\) .

The diffusion framework's flexibility stems largely from its ability to incorporate guidance mechanisms, which afford steering the sampling process toward desired outcomes by including additional information or constraints. Notable examples include classifier guidance (Dhariwal & Nichol, 2021) and classifier- free guidance (Ho & Salimans, 2022), which enable variable- strength conditioned generation. For inverse problems, guidance methods exist to incorporate information about observations (Chung et al., 2023; Song et al., 2023a).

### 2.2 DIFFUSION-BASED AMORTIZED SBI

Modern amortized SBI methods leverage highly expressive generative models for multiple inference tasks. Early applications of diffusion models to SBI include Geffner et al. (2023), who demonstrated that learning conditional score functions via denoising score matching offers superior stability and sample quality compared to traditional flow- based baselines. Furthermore, Simformer (Gloeckler et al., 2024) trains a diffusion model on samples from the joint distribution \(p(\pmb {\theta}, \mathbf{x}) = p_{\mathrm{train}}(\pmb {\theta}) p(\mathbf{x} | \pmb {\theta})\) . Simformer employs a transformer architecture to model the score function \(s_{\phi}(\pmb {\xi}_t, t, \mathrm{mask})\) of the noised joint variable \(\pmb {\xi}_t = (\pmb {\theta}_t, \mathbf{x}_t)\) at diffusion time \(t\) . The mask specifies which components of \(\pmb{\xi}\) are conditioned upon and which are to be generated. By setting the mask appropriately (e.g., conditioning on \(\mathbf{x}\) to generate \(\pmb {\theta}\) ), Simformer can sample from various conditionals, including the posterior \(p(\pmb {\theta} | \mathbf{x})\) and posterior predictive \(p(\mathbf{x}^* | \mathbf{x})\) . Crucially, these learned conditionals are implicitly tied to the training prior \(p_{\mathrm{train}}(\pmb {\theta})\) used to generate the training data—applying a different prior \(q(\pmb {\theta})\) would require retraining the diffusion model with the new prior.

### 2.3 PRIOR ADAPTATION IN AMORTIZED SBI

Standard amortized SBI methods tie the learned posterior \(q_{\phi}(\pmb {\theta} | \mathbf{x})\) to the fixed prior \(p_{\mathrm{train}}(\pmb {\theta})\) used during training. Changing this prior traditionally requires retraining the model from scratch, which is computationally prohibitive for expensive simulators. A few recent amortized methods achieve prior flexibility by training over a meta- prior—a distribution or predefined set of possible prior specifications—to learn how to incorporate different prior information at runtime. For instance, the Amortized Conditioning Engine (ACE; Chang et al., 2025) allows users to specify factorized priors at runtime by encoding each one- dimensional prior density as a normalized histogram over a predefined grid; its transformer architecture is trained to process these specific histogram- based prior encodings alongside observed data. Similarly, the Distribution Transformer (DT; Whittle et al., 2025) learns a direct mapping from a prior, itself represented as a Gaussian mixture model (GMM), to a GMM posterior, using attention mechanisms to transform the prior components based on the data. Sensitivity- aware SBI (Elsemüller et al., 2024) focuses on providing efficient sensitivity analysis to various modeling choices, including different priors, represented by a discrete set of possible alternative prior specifications. All of these methods enable prior adaptation by relying on their pre- training across a chosen meta- prior. Our approach, described next, sidesteps this requirement by performing purely inference- time adaptation to flexibly handle new target priors.

## 3 PRIOR GUIDE

PriorGuide offers a solution to take a diffusion- based amortized SBI model such as Simformer (Gloeckler et al., 2024), trained on a training prior \(p_{\mathrm{train}}(\pmb {\theta})\) , and adapt it to a new target prior \(q(\pmb {\theta})\) at inference time, without retraining. The objective is to perform standard SBI tasks such as sampling from the posterior or posterior predictive distribution under the new prior, that is \(q(\pmb {\theta} | \mathbf{x}) \propto p(\mathbf{x} | \pmb {\theta}) q(\pmb {\theta})\) or \(q(\mathbf{x}^* | \mathbf{x})\) , respectively. The method achieves this by adjusting the score guidance during the diffusion sampling process. The key relationship for this adaptation is as follows:

Proposition 1. Let the posterior under the original prior be \(p(\pmb {\theta} | \mathbf{x}) \propto p_{\mathrm{train}}(\pmb {\theta}) p(\mathbf{x} | \pmb {\theta})\) , and let the target posterior—the posterior under the new prior—be \(q(\pmb {\theta} | \mathbf{x}) \propto q(\pmb {\theta}) p(\mathbf{x} | \pmb {\theta})\) . Then, sampling from \(q(\pmb {\theta} | \mathbf{x})\) is equivalent to sampling from \(r(\pmb {\theta}) p(\pmb {\theta} | \mathbf{x})\) with \(r(\pmb {\theta}) \equiv \frac{q(\pmb{\theta})}{p_{\mathrm{train}}(\pmb{\theta})}\) the prior ratio.

Proof. We can rewrite the target posterior \(q(\theta |\mathbf{x})\) as

\[q(\pmb {\theta}|\mathbf{x})\propto q(\pmb {\theta})p(\mathbf{x}|\pmb {\theta}) = \frac{q(\pmb{\theta})}{p_{\mathrm{train}}(\pmb{\theta})} p_{\mathrm{train}}(\pmb {\theta})p(\mathbf{x}|\pmb {\theta})\propto \frac{q(\pmb{\theta})}{p_{\mathrm{train}}(\pmb{\theta})} p(\pmb {\theta}|\mathbf{x}) = r(\pmb {\theta})p(\pmb {\theta}|\mathbf{x}),\]

where the prior ratio \(r(\pmb {\theta})\equiv \frac{q(\pmb{\theta})}{p_{\mathrm{train}}(\pmb{\theta})}\) takes the role of an importance weighing function, analogous to the correction applied in multi- round neural posterior estimation (Lueckmann et al., 2017). \(\square\)

Next, we focus on the task of sampling from the target posterior. First, we show in Section 3.1 how introducing the new prior amounts to adding a guidance term to the diffusion process for posterior sampling. In Section 3.2, we develop analytical approximations to make this tractable. Section 3.3 shows how we can provide guarantees using corrective Langevin steps. Finally, in Section 3.4 we show how our results for posterior sampling readily extend to the posterior predictive case.

### 3.1 TARGET PRIOR AS GUIDANCE

Assume we have a diffusion model trained under \(p_{\mathrm{train}}(\pmb {\theta})\) to sample from \(p(\pmb {\theta}|\mathbf{x})\) with learnt score model \(s(\pmb {\theta}_t,t,\mathbf{x})\) . Prior Guide leverages the fact that we can relate the score of the target posterior \(q(\pmb {\theta}|\mathbf{x})\) to the original score. The marginal pdf at time \(t\) of the diffusion process for \(q(\pmb {\theta}|\mathbf{x})\) is:

\[q(\pmb {\theta}_t|\mathbf{x})\propto \int r(\pmb {\theta}_0)p(\pmb {\theta}_0|\mathbf{x})p(\pmb {\theta}_t|\pmb {\theta}_0)\mathrm{d}\pmb {\theta}_0, \quad (5)\]

which is written as an integral over \(\pmb{\theta}_0\) by noting that \(q(\pmb {\theta}_0|\mathbf{x})\propto r(\pmb {\theta}_0)p(\pmb {\theta}_0|\mathbf{x})\) and then we propagated this to time \(t\) by convolution with the transition kernel \(p(\pmb {\theta}_t|\pmb {\theta}_0)\) . Thus, the score is:

\[\begin{array}{r l} & {\nabla_{\pmb{\theta}_{t}}\log q(\pmb{\theta}_{t}|\mathbf{x}) = \nabla_{\pmb{\theta}_{t}}\log \int r(\pmb{\theta}_{0})p(\pmb{\theta}_{0}|\mathbf{x})p(\pmb{\theta}_{t}|\pmb{\theta}_{0},\mathbf{x})\mathrm{d}\pmb{\theta}_{0} \quad (6)}\\ & {\qquad = \nabla_{\pmb{\theta}_{t}}\log \int r(\pmb{\theta}_{0})p(\pmb{\theta}_{0}|\pmb{\theta}_{t},\mathbf{x})p(\pmb{\theta}_{t}|\mathbf{x})\mathrm{d}\pmb{\theta}_{0} \quad (7)}\\ & {\qquad = \nabla_{\pmb{\theta}_{t}}\log p(\pmb{\theta}_{t}|\mathbf{x}) + \nabla_{\pmb{\theta}_{t}}\log \int r(\pmb{\theta}_{0})p(\pmb{\theta}_{0}|\pmb{\theta}_{t},\mathbf{x})\mathrm{d}\pmb{\theta}_{0} \quad (8)}\\ & {\qquad = \underbrace{s(\pmb{\theta}_{t},t,\mathbf{x})}_{\mathrm{original~score}} + \nabla_{\pmb{\theta}_{t}}\log \underbrace{\mathbb{E}_{p(\pmb{\theta}_{0}|\pmb{\theta}_{t},\mathbf{x})}}_{\mathrm{reverse~kernel}}\left[\underbrace{r(\pmb{\theta}_{0})}_{\mathrm{prior~ratio}}\right], \quad (9)} \end{array}\]

where in Eq. (7) we re- express the joint probability \(p(\pmb {\theta}_0|\mathbf{x})p(\pmb {\theta}_t|\pmb {\theta}_0,\mathbf{x}) = p(\pmb {\theta}_0,\pmb {\theta}_t|\mathbf{x})\) as \(p(\pmb {\theta}_0|\pmb {\theta}_t,\mathbf{x})p(\pmb {\theta}_t|\mathbf{x})\) , which allows us to separate the contribution of the new prior guidance from the original score model \(s(\pmb {\theta}_t,t,\mathbf{x})\) . In multiple steps we exploit the fact that multiplicative constants inside the integral disappear under the score. In conclusion, Eq. (9) expresses the score of the target (new) posterior as the old score, which we have, plus a guidance term which we can estimate.

Guided diffusion We can draw samples from the posterior distribution via the reverse diffusion process using the modified score in Eq. (9). The first term is the trained score model and the second term estimates how the new prior's influence propagates to time \(t\) (guidance term). This is a common way to implement a guidance function (Chung et al., 2023; Song et al., 2023a;b; Rissanen et al., 2025), which now depends on the prior ratio. The core challenge lies in evaluating the expectation over \(\pmb{\theta}_0\) , which is intractable and requires simulating the reverse SDE. To make this tractable, we develop analytical approximations in the following section.

Ensuring prior coverage A crucial consideration for stable guidance is ensuring the new prior \(q(\pmb {\theta})\) remains within regions adequately covered by the training prior \(p_{\mathrm{train}}(\pmb {\theta})\) .2 If \(q(\pmb {\theta})\) assigns significant mass to regions where \(p_{\mathrm{train}}(\pmb {\theta})\) has negligible support (i.e., is out- of- distribution or OOD; Lee et al., 2018; Nalisnick et al., 2019), two related issues can arise: (a) in these regions, the learned score \(s(\pmb {\theta}_t,t,\mathbf{x})\) is likely a poor representation of the true score, as the training set contained few examples in these regions; and (b) the prior ratio \(r(\pmb {\theta}) = q(\pmb {\theta}) / p_{\mathrm{train}}(\pmb {\theta})\) can become arbitrarily large or ill- defined, destabilizing the guidance mechanism. Lack of coverage can be quantified using OOD metrics (Lee et al., 2018; Nalisnick et al., 2019; Schmitt et al., 2023; Huang et al., 2024). Notably, the requirement that \(q(\pmb {\theta})\) should be covered by \(p_{\mathrm{train}}(\pmb {\theta})\) is typically not restrictive since the common practice is to train amortized models on broad training priors. This diagnostic check is detailed in Appendix A.4.

### 3.2 APPROXIMATING THE GUIDANCE FUNCTION

To approximate the guidance term in Eq. (9) efficiently while maintaining flexible test- time priors, we introduce two approximations. Following recent work (Song et al., 2023a; Peng et al., 2024; Rissanen et al., 2025), we model the reverse transition kernel as a Gaussian. We then introduce a novel approach that represents \(r(\pmb {\theta})\) as a Gaussian mixture model. This yields an analytical solution for the guidance, circumventing the issue of estimating the score of an expectation via Monte Carlo, which would suffer from both bias and variance.

Reverse transition kernel approximation We first approximate the reverse transition kernel \(p(\pmb{\theta}_{0}\mid \pmb{\theta}_{t},\mathbf{x})\) as a multivariate Gaussian distribution:

\[p(\pmb{\theta}_{0}\mid \pmb{\theta}_{t},\mathbf{x})\approx \mathcal{N}\left(\pmb{\theta}_{0}\mid \pmb{\mu}_{0\mid t}(\pmb{\theta}_{t},\mathbf{x}),\pmb{\Sigma}_{0\mid t}\right) \quad (10)\]

whose mean is obtained from the score function via Tweedie's formula (Song & Ermon, 2019):

\[\pmb{\mu}_{0\mid t}(\pmb{\theta}_{t},\mathbf{x}) = \pmb{\theta}_{t} + \sigma (t)^{2}\nabla_{\pmb{\theta}_{t}}\log p(\pmb{\theta}_{t}\mid \mathbf{x}). \quad (11)\]

This approximation is common in the guidance literature (Chung et al., 2023; Song et al., 2023a; Boys et al.; Peng et al., 2024; Rissanen et al., 2025; Finzi et al., 2023; Bao et al., 2022). For the covariance matrix \(\pmb{\Sigma}_{0\mid t}\) , we adopt a simple yet effective approximation inspired by (Song et al., 2023a; Ho et al., 2022):

\[\pmb{\Sigma}_{0\mid t} = \frac{\sigma(t)^{2}}{1 + \sigma(t)^{2}}\mathbf{I}. \quad (12)\]

This approximation acts as a time- dependent scaling factor that naturally aligns with the diffusion process—starting at the identity matrix when \(t = 1\) and approaching zero as \(t \to 0\) , effectively increasing the precision of our prior guidance at smaller timesteps. This approximation becomes exact for all \(t\) if the posterior under the original target distribution is \(p(\pmb{\theta}_{0}\mid \mathbf{x}) = \mathcal{N}(\pmb{\theta}_{0}\mid \mathbf{0},\mathbf{I})\) .

Prior ratio approximation With the goal of obtaining a closed- form solution for the guidance, we then approximate the prior ratio function \(r(\pmb {\theta}) = \frac{q(\pmb{\theta})}{p(\pmb{\theta})}\) as a generalized mixture of Gaussians:

\[r(\pmb {\theta})\approx \sum_{i = 1}^{K}w_{i}\mathcal{N}(\pmb {\theta}\mid \pmb{\mu}_{i},\pmb{\Sigma}_{i}),\qquad r(\pmb {\theta})\geq 0, \quad (13)\]

where \(\{w_{i},\pmb{\mu}_{i},\pmb{\Sigma}_{i}\}_{i = 1}^{K}\) represent the weights, means and covariance matrices of the mixture, with \(K\) a hyperparameter denoting the number of mixture components. Since this represents a ratio rather than a distribution, the mixture weights need not be positive nor sum to one, as long as the ratio remains non- negative, potentially enabling more expressive approximations such as subtractive mixtures (Loconte et al., 2024). Notably, when \(p(\pmb {\theta})\) is uniform, \(r(\pmb {\theta})\propto q(\pmb {\theta})\) . Since the guidance term is a gradient of a log- expectation, the constant factor vanishes, allowing us to directly specify \(q(\pmb {\theta})\) as a Gaussian mixture. For the more general case of non- uniform training distributions, obtaining the Gaussian mixture approximation for the ratio is a standard function approximation task (Sorenson & Alspach, 1971). Crucially, since the densities of both \(p_{\mathrm{train}}\) and \(q\) are analytically known, fitting the ratio avoids the instability and high variance inherent to statistical density- ratio estimation from finite samples. We provide a straightforward gradient- based fitting procedure in Appendix A.2.

Guidance term Plugging in Eq. (10) and Eq. (13), the guidance from Eq. (9) becomes:

\[\nabla_{\pmb{\theta}_{t}}\log \mathbb{E}_{p(\pmb{\theta}_{0}\mid \pmb{\theta}_{t},\mathbf{x})}\left[r(\pmb{\theta}_{0})\right]\approx \nabla_{\pmb{\theta}_{t}}\log \int \sum_{i = 1}^{K}w_{i}\mathcal{N}(\pmb {\theta}_{0}\mid \pmb{\mu}_{i},\pmb{\Sigma}_{i})\mathcal{N}(\pmb {\theta}_{0}\mid \pmb{\mu}_{0\mid t}(\pmb {\theta}_{t},\mathbf{x}),\pmb{\Sigma}_{0\mid t})\mathrm{d}\pmb{\theta}_{0}. \quad (14)\]

This integral can be solved analytically (full derivation in Appendix B), yielding:

\[\nabla_{\pmb{\theta}_{t}}\log \mathbb{E}_{p(\pmb{\theta}_{0}\mid \pmb{\theta}_{t},\mathbf{x})}[r(\pmb {\theta}_{0})]\approx \sum_{i = 1}^{K}\tilde{w}_{i}(\pmb{\mu}_{i} - \pmb{\mu}_{0\mid t}(\pmb{\theta}_{t}))^{\top}\tilde{\pmb{\Sigma}}_{i}^{-1}\nabla_{\pmb{\theta}_{t}}\pmb{\mu}_{0\mid t}(\pmb{\theta}_{t}), \quad (15)\]

where \(\widetilde{\mathbf{S}}_{i} = \pmb{\Sigma}_{i} + \pmb{\Sigma}_{0|t}\) and \(\tilde{w}_{i} = w_{i}\mathcal{N}(\pmb{\mu}_{i}\mid \pmb{\mu}_{0|t}(\pmb{\theta}_{t},\mathbf{x}),\widetilde{\mathbf{S}}_{i}) / \sum_{j = 1}^{K}w_{j}\mathcal{N}(\pmb{\mu}_{j}\mid \pmb{\mu}_{0|t}(\pmb{\theta}_{t},\mathbf{x}),\widetilde{\mathbf{S}}_{j})\) .

Finally, the PriorGuide update to the mean of the reverse kernel can be expressed concisely using Tweedie's formula, Eq. (11), and our derived guidance term, Eq. (15):

\[\mu_{0|t}^{\mathrm{new}}(\pmb{\theta}_{t},\mathbf{x}) = \mu_{0|t}(\pmb{\theta}_{t},\mathbf{x}) + \sigma (t)^{2}\sum_{i}^{K}\tilde{w}_{i}(\pmb{\mu}_{i} - \pmb{\mu}_{0|t}(\pmb{\theta}_{t},\mathbf{x}))^{\top}\widetilde{\mathbf{\Sigma}}_{i}^{-1}\nabla_{\pmb{\theta}_{t}}\pmb{\mu}_{0|t}(\pmb{\theta}_{t},\mathbf{x}). \quad (16)\]

This update intuitively combines the original prediction \(\mu_{0|t}(\pmb{\theta}_{t})\) based on the training prior with a weighted sum of correction terms from our new prior. The correction magnitude is controlled by both the noise schedule \(\sigma (t)^{2}\) and the distance between the mixture components and current prediction.

### 3.3 ASYMPTOTICALLY CORRECT SAMPLING WITH LANGEVIN DYNAMICS

The diffusion sampling process detailed in Section 3.2 is approximate due to the Gaussian approximation of the reverse transition kernel \(p(\pmb{\theta}_{0}\mid \pmb{\theta}_{t},\mathbf{x})\) . However, as stated in the following proposition, this Gaussian approximation becomes correct on low noise levels, rendering our guidance term accurate.

Proposition 2. As \(t,\sigma (t)\to 0\) , the approximation \(\mathcal{N}(\pmb{\theta}_{0}\mid \pmb{\theta}_{t} + \sigma (t)^{2}\nabla_{\pmb{\theta}_{t}}\log p(\pmb{\theta}_{t}),\frac{\sigma(t)^{2}}{1 + \sigma(t)^{2}}\mathbf{I})\) converges to the true \(p(\pmb{\theta}_{0}\mid \pmb{\theta}_{t})\) , under mild regularity conditions on \(p(\pmb{\theta}_{0})\) .

We provide the exact statement and a proof in Appendix B. Close- by statements are well- known in the diffusion literature (Finzi et al., 2023; Ho et al., 2022). Thus, the guidance approximation is correct at low noise levels (up to the GMM prior ratio fit accuracy), and we can run accurate Langevin dynamics MCMC sampling (Sarkka & Solin, 2019). To incorporate this with the regular diffusion process, we run \(N_{L}\) Langevin steps after each regular diffusion step, effectively transforming the sampling into an annealed MCMC process that resembles methods used in compositional generation (Geffner et al., 2023; Du et al., 2023) and early unconditional diffusion models (Song et al., 2021). This refinement increases the total inference cost by a factor of \(N_{L} + 1\) .

Thus, PriorGuide has two main hyperparameters: the number of diffusion steps \(N > 0\) , as per any diffusion model, and the number of interleaved Langevin steps \(N_{L} \geq 0\) . For test- time sampling, the total number of function evaluations (NFE), or forward passes of the trained network, is \(N \times (N_{L} + 1)\) .

### 3.4 PRIORGUIDE POSTERIOR PREDICTIVE SAMPLING

PriorGuide is readily applied to compute posterior predictive distributions as well under a new prior. Starting from a diffusion model trained to generate samples from the joint posterior predictive distribution, \(p(\mathbf{x}^{*},\pmb {\theta}\mid \mathbf{x})\) , we can marginalize over \(\pmb{\theta}\) to get the posterior predictive \(p(\mathbf{x}^{*}\mid \mathbf{x})\) . The joint posterior predictive under the new prior becomes:

\[q(\mathbf{x}^{*},\pmb {\theta}\mid \mathbf{x}) = q(\mathbf{x}^{*}\mid \pmb {\theta},\mathbf{x})q(\pmb {\theta}\mid \mathbf{x})\propto q(\mathbf{x}^{*}\mid \pmb {\theta},\mathbf{x})r(\pmb {\theta})p(\pmb {\theta}\mid \mathbf{x}) = r(\pmb {\theta})p(\mathbf{x}^{*},\pmb {\theta}\mid \mathbf{x}),\]

which results in a posterior predictive version of Eq. (9):

\[\nabla_{\mathbf{x}_{t}^{*},\pmb{\theta}_{t}}\log q(\mathbf{x}_{t}^{*},\pmb{\theta}_{t}\mid \mathbf{x}) = s(\mathbf{x}_{t}^{*},\pmb{\theta}_{t},t,\mathbf{x}) + \nabla_{\mathbf{x}_{t}^{*},\pmb{\theta}_{t}}\log \mathbb{E}_{p(\pmb{\theta}_{0}\mid \mathbf{x}_{t}^{*},\pmb{\theta}_{t},\mathbf{x})}\left[r(\pmb{\theta}_{0})\right]. \quad (17)\]

The posterior predictive and posterior guidance terms differ only in the conditioning information for the score and reverse transition kernel. Thus, everything presented earlier in this section applies to this scenario, and the posterior predictive is obtained from the previous formulas with substitutions \(p(\pmb{\theta}_{0}\mid \pmb{\theta}_{t},\mathbf{x})\to p(\pmb{\theta}_{0}\mid \pmb{\xi}_{t}^{*},\mathbf{x})\) , \(\mu_{0|t}(\pmb{\theta}_{t},\mathbf{x})\to \mu_{0|t}(\pmb{\xi}_{t}^{*},\mathbf{x})\) and \(\nabla_{\pmb{\theta}}\to \nabla_{\pmb{\xi}^{*}}\) , where \(\pmb{\xi}_{t}^{*}\equiv (\mathbf{x}_{t}^{*},\pmb{\theta}_{t})\) .

## 4 EXPERIMENTS

We empirically evaluate PriorGuide across a range of SBI problems, focusing on its ability to adapt to new priors at test time for posterior and posterior predictive inference. First, Section 4.1 provides an intuitive demonstration on a 2D problem. In Section 4.2, we evaluate posterior inference on several SBI problems, comparing PriorGuide to existing methods that support test- time prior adaptation. Section 4.3 examines PriorGuide's performance on challenging posterior predictive tasks. Finally, Section 4.4 studies the trade- off between computational cost and inference accuracy. Full experimental details can be found in Appendix C, with Appendix D.3 reporting additional baseline

results. We also conduct two ablation studies, including an analysis of the sensitivity of PriorGuide to the distance between training and test- time priors (Appendix D.4) and a study of the impact of the number of GMM components used to approximate the prior ratio (Appendix D.5). The code is available at https://github.com/acerbilab/prior- guide.

### 4.1 ILLUSTRATIVE EXAMPLE OF TEST-TIME PRIOR ADAPTATION

We illustrate PriorGuide's capabilities on Two Moons, a two- dimensional SBI model with a bimodal posterior (Greenberg et al., 2019). We train the diffusion model under a uniform prior \(p_{\mathrm{train}}(\pmb {\theta})\) over \([- 1,1]^2\) , and test how PriorGuide handles a new prior \(q(\pmb {\theta})\) at test time. Fig. A1 shows that PriorGuide incorporates the new prior, matching well the true Bayesian posterior under the new prior.

### 4.2 TEST-TIME PRIOR ADAPTATION FOR POSTERIOR INFERENCE

We evaluate PriorGuide's posterior inference capabilities on six SBI problems (see Table A1), ranging from established SBI benchmarks to real models from engineering and neuroscience: Two Moons (Lueckmann et al., 2021); the Ornstein- Uhlenbeck Process (OUP; Uhlenbeck & Ornstein, 1930); the Turin model of radio propagation (Turin et al., 1972); the Gaussian Linear model (Lueckmann et al., 2021) and its high- dimensional variant; and the Bayesian Causal Inference model of multisensory perception (BCI; Kording et al., 2007). Training priors \(p_{\mathrm{train}}(\pmb {\theta})\) for the base diffusion model (Simformer; Gloeckler et al., 2024) were uniform or Gaussian; details in Appendix C.

For baselines, we consider the base Simformer (no prior adaptation) and the Amortized Conditioning Engine (ACE; Chang et al., 2025), one of several approaches (Elsemüller et al., 2024; Whittle et al., 2025) that amortizes test- time prior adaptation for posterior inference by pre- training on a variety of possible (factorized) priors. PriorGuide is more flexible than ACE by not needing pretraining on specific priors—instead, it modifies a diffusion- based amortized inference model at runtime—, and can represent correlated and non- factorized priors. Detailed comparisons against additional non- amortized methods—including classic algorithms (rejection sampling and sampling- importance- resampling) and neural likelihood estimation (NLE; Papamakarios et al., 2019) with MCMC—are provided in Appendix D.3. While often computationally expensive, these methods serve as fundamental benchmarks for posterior sampling.

We consider three different families of target priors: mild, strong and mixture. Mild and strong priors are defined as multivariate Gaussian distributions with means drawn from a uniform box and diagonal covariance matrices, where the strong priors have smaller standard deviations; these represent scenarios with varying degrees of available information. Mixture priors are defined as a mixture distribution with two multivariate Gaussian components with the same setup as the strong priors; this can represent situations with distinct, competing hypotheses about the parameter values. For each prior family, we randomly generate ten possible prior parameterizations \(q^{(i)}(\pmb {\theta})\) , sample ten parameter vectors \(\pmb {\theta}_{i,j} \sim q^{(i)}(\pmb {\theta})\) from that prior, and simulate one observed dataset per \(\pmb{\theta}_{i,j}\) , \(\mathbf{x}_{i,j} \sim p(\mathbf{x} \mid \pmb{\theta}_{i,j})\) to evaluate the methods. See Appendix C.3.1 for the full procedure.

We measure each method's performance using: 1) the root mean squared error (RMSE) between the true parameter and the samples from the estimated posterior; 2) the classifier two- sample test (C2ST) between the estimated posterior samples and ground- truth posterior samples; 3) the mean marginal total variation distance (MMTV) between the estimated vs. ground- truth posterior samples. For RMSE and MMTV, lower is better, while for C2ST, closer to 0.5 is better.

Results in Table 1 show that PriorGuide largely improves inference accuracy over the base Simformer model in all scenarios, making use of the prior information provided at test time, and achieves leading performance in most cases, especially when stronger prior beliefs (strong and mixture) are presented.

### 4.3 TEST-TIME PRIOR ADAPTATION FOR DATA PREDICTION

We next evaluate PriorGuide's ability to perform posterior predictive inference under new target priors, focusing on forecasting or retrocasting scenarios, as shown in Fig. 1. We use the OUP and Turin models, both of which generate time series trajectories (Fig. 2). We employ the same procedure and test- time prior setup (mild, strong, and mixture) from Section 4.2. For each target prior, we

Table 1: Posterior inference \((\theta)\) . Mean (standard dev.) over 5 independent training runs (10 random target priors \(\times 10\) simulated datasets). Significantly best results (Wilcoxon signed-rank test) in bold.   

| Task | Method | RMSE ($q_{\mathrm{mild}}$) | C2ST ($q_{\mathrm{mild}}$) | MMTV ($q_{\mathrm{mild}}$) | RMSE ($q_{\mathrm{strong}}$) | C2ST ($q_{\mathrm{strong}}$) | MMTV ($q_{\mathrm{strong}}$) | RMSE ($q_{\mathrm{mixture}}$) | C2ST ($q_{\mathrm{mixture}}$) | MMTV ($q_{\mathrm{mixture}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Two Moons | Simformer | 0.39 (0.19) | 0.56 (0.05) | 0.21 (0.10) | 0.39 (0.20) | 0.75 (0.06) | 0.54 (0.13) | 0.33 (0.22) | 0.68 (0.08) | 0.42 (0.17) |
| Two Moons | ACE | 0.34 (0.16) | 0.83 (0.04) | 0.23 (0.07) | 0.09 (0.03) | 0.79 (0.05) | 0.35 (0.12) | 0.16 (0.15) | 0.80 (0.07) | 0.34 (0.14) |
| Two Moons | PriorGuide | 0.37 (0.16) | 0.52 (0.02) | 0.11 (0.04) | 0.09 (0.04) | 0.52 (0.02) | 0.08 (0.02) | 0.20 (0.17) | 0.55 (0.05) | 0.16 (0.11) |
| OUP | Simformer | 0.18 (0.07) | 0.58 (0.08) | 0.18 (0.11) | 0.24 (0.08) | 0.73 (0.08) | 0.37 (0.10) | 0.23 (0.09) | 0.70 (0.08) | 0.33 (0.11) |
| OUP | ACE | 0.17 (0.07) | 0.58 (0.03) | 0.11 (0.05) | 0.14 (0.04) | 0.56 (0.03) | 0.12 (0.05) | 0.17 (0.07) | 0.62 (0.08) | 0.20 (0.11) |
| OUP | PriorGuide | 0.17 (0.06) | 0.52 (0.02) | 0.08 (0.04) | 0.13 (0.04) | 0.51 (0.02) | 0.06 (0.02) | 0.15 (0.06) | 0.51 (0.02) | 0.07 (0.04) |
| Turin | Simformer | 0.22 (0.03) | 0.80 (0.03) | 0.31 (0.04) | 0.23 (0.03) | 0.95 (0.02) | 0.56 (0.07) | 0.23 (0.03) | 0.94 (0.02) | 0.50 (0.06) |
| Turin | ACE | 0.21 (0.02) | 0.78 (0.03) | 0.27 (0.04) | 0.17 (0.02) | 0.92 (0.04) | 0.47 (0.07) | 0.19 (0.03) | 0.91 (0.03) | 0.44 (0.06) |
| Turin | PriorGuide | 0.14 (0.02) | 0.64 (0.06) | 0.13 (0.05) | 0.06 (0.01) | 0.55 (0.04) | 0.08 (0.03) | 0.13 (0.06) | 0.62 (0.02) | 0.19 (0.12) |
| Gaussian Linear 10D | Simformer | 0.29 (0.02) | 0.89 (0.02) | 0.30 (0.03) | 0.31 (0.03) | 1.00 (0.00) | 0.65 (0.03) | 0.30 (0.03) | 0.99 (0.00) | 0.61 (0.05) |
| Gaussian Linear 10D | ACE | 0.22 (0.02) | 0.67 (0.04) | 0.12 (0.02) | 0.10 (0.01) | 0.78 (0.06) | 0.19 (0.04) | 0.19 (0.05) | 0.96 (0.03) | 0.40 (0.11) |
| Gaussian Linear 10D | PriorGuide | 0.31 (0.08) | 0.53 (0.03) | 0.05 (0.01) | 0.23 (0.09) | 0.54 (0.05) | 0.06 (0.02) | 0.26 (0.09) | 0.57 (0.06) | 0.12 (0.06) |
| Gaussian Linear 20D | Simformer | 0.29 (0.02) | 0.95 (0.01) | 0.29 (0.02) | 0.30 (0.02) | 1.00 (0.00) | 0.64 (0.02) | 0.30 (0.02) | 1.00 (0.00) | 0.63 (0.03) |
| Gaussian Linear 20D | ACE | 0.22 (0.02) | 0.72 (0.06) | 0.11 (0.02) | 0.10 (0.01) | 0.82 (0.05) | 0.16 (0.03) | 0.20 (0.04) | 0.99 (0.01) | 0.44 (0.09) |
| Gaussian Linear 20D | PriorGuide | 0.27 (0.07) | 0.54 (0.03) | 0.05 (0.01) | 0.28 (0.12) | 0.58 (0.03) | 0.11 (0.03) | 0.28 (0.10) | 0.59 (0.05) | 0.14 (0.06) |
| BCI | Simformer | 0.79 (0.13) | 0.86 (0.05) | 0.35 (0.09) | 1.03 (0.16) | 0.98 (0.01) | 0.61 (0.09) | 0.99 (0.24) | 0.98 (0.02) | 0.63 (0.09) |
| BCI | ACE | 0.55 (0.12) | 0.84 (0.08) | 0.28 (0.10) | 0.31 (0.13) | 0.82 (0.09) | 0.29 (0.12) | 1.02 (0.37) | 0.97 (0.02) | 0.56 (0.11) |
| BCI | PriorGuide | 0.56 (0.10) | 0.88 (0.06) | 0.41 (0.08) | 0.25 (0.04) | 0.72 (0.10) | 0.21 (0.12) | 0.87 (0.68) | 0.78 (0.15) | 0.37 (0.27) |

> **Image description.** The image is a technical figure consisting of two side-by-side line graphs, illustrating posterior predictive distributions for two different models, OUP and Turin. The figure is titled "Figure 2: Example posterior predictive distributions for OUP and Turin models (strong priors)."
>
> **General Layout and Legend:**
> Both graphs share a legend positioned above the plots, which identifies various models and methods: Simformer (pink), PriorGuide (blue), ACE (green), OUP (light blue/cyan), and Turin (dark blue).
>
> **Left Panel (OUP Model):**
> This graph is titled "OUP" and displays the "Process value" on the vertical (Y) axis, ranging from 0 to 10. The horizontal (X) axis represents the "Time step," ranging from 0 to 24. The data shows multiple colored lines representing the predictions of different models. All lines start at high process values (near 10) at Time step 0 and exhibit a general downward trend, stabilizing or fluctuating between 2 and 5 as the time step increases toward 24.
>
> **Right Panel (Turin Model):**
> This graph is titled "Turin" and displays "Power" on the vertical (Y) axis, ranging from 0.0 to 1.0. The horizontal (X) axis represents the "Time step," ranging from 0 to 100. Similar to the left panel, multiple colored lines are plotted. These lines generally start at high power values (near 1.0) and show a gradual decrease over the 100 time steps, with some fluctuations and stabilization at lower power levels (between 0.2 and 0.6).
>
> The overall visual presentation uses thin, distinct colored lines to differentiate the performance of the various predictive models across the two different scenarios (OUP and Turin).

<center>Figure 2: Example posterior predictive distributions for OUP and Turin models (strong priors). </center>

condition the model on partial trajectories: in half of the cases, the first \(30\%\) of the trajectory, and for the other half, the last \(30\%\) . The task is always to predict the unobserved \(70\%\) of the trajectory.

We evaluate the performance of all methods using RMSE and the maximum mean discrepancy (MMD) with an exponentiated quadratic kernel between the ground- truth trajectory \(\mathbf{x}_o\) and generated posterior predictive samples. Fig. 2 shows how example posterior predictive distributions from PriorGuide closely match the true data. Results in Table 2 show that PriorGuide can generate reliable posterior predictive samples and achieve performance on par with or better than other methods.

Table 2: Data prediction (x). Mean (standard dev.) over 5 independent training runs (10 random target priors \(\times 10\) simulated datasets). Significantly best results (Wilcoxon signed-rank test) in bold.   

| Task | Method | RMSE ($q_{\mathrm{mild}}$) | MMD$_x$ ($q_{\mathrm{mild}}$) | RMSE ($q_{\mathrm{strong}}$) | MMD$_x$ ($q_{\mathrm{strong}}$) | RMSE ($q_{\mathrm{mixture}}$) | MMD$_x$ ($q_{\mathrm{mixture}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OUP | Simformer | 0.32 (0.10) | 0.44 (0.24) | 0.39 (0.11) | 0.54 (0.24) | 0.34 (0.11) | 0.47 (0.25) |
| OUP | ACE | 0.26 (0.08) | 0.44 (0.31) | 0.22 (0.06) | 0.30 (0.20) | 0.24 (0.10) | 0.38 (0.31) |
| OUP | PriorGuide | 0.28 (0.09) | 0.45 (0.28) | 0.21 (0.05) | 0.29 (0.17) | 0.25 (0.11) | 0.34 (0.22) |
| Turin | Simformer | 0.14 (0.01) | 0.49 (0.09) | 0.14 (0.01) | 0.49 (0.09) | 0.14 (0.01) | 0.48 (0.09) |
| Turin | ACE | 0.16 (0.03) | 0.62 (0.19) | 0.16 (0.03) | 0.61 (0.20) | 0.16 (0.03) | 0.61 (0.19) |
| Turin | PriorGuide | 0.13 (0.01) | 0.47 (0.08) | 0.13 (0.01) | 0.46 (0.07) | 0.13 (0.01) | 0.46 (0.09) |

### 4.4 TEST-TIME REFINEMENT VIA CORRECTIVE LANGEVIN DYNAMICS

PriorGuide supports improving the sampling quality by adding Langevin dynamic steps to the diffusion process, at the cost of additional test- time compute. We examine posterior inference accuracy—measured by MMTV, but similar results hold for other metrics—on the OUP and Turin models as a function of the number of diffusion steps \(N\) and Langevin steps \(N_{L}\) . These two can be combined into a single computational cost metric, the number of function evaluations (NFEs), i.e.

> **Image description.** This image consists of two side-by-side technical line graphs, labeled (a) and (b), which illustrate the relationship between MMTV (Mean Marginal Total Variation) and the Number of Function Evaluations (NFEs) for posterior inference under strong priors.
>
> **General Structure and Shared Elements:**
> Both panels are scatter plots with connected lines, displaying performance metrics across varying computational budgets.
> *   **Y-axis:** Labeled "MMTV," representing the performance metric.
> *   **X-axis:** Labeled "Number of Function Evaluations," with major tick marks at 100 and 1000.
> *   **Pareto Frontier:** Both graphs feature a dashed black line labeled "Pareto Frontier," which slopes downward, indicating the optimal trade-off between MMTV and NFEs.
> *   **Data Representation:** Data points are color-coded and shaped according to the "Diffusion Steps" used in the inference process.
>
> **Panel (a): Posterior inference on OUP, strong priors**
> *   **Title:** (a) Posterior inference on OUP, strong priors.
> *   **Y-axis Range:** MMTV ranges from approximately 0.04 to 0.08.
> *   **Legend (Diffusion Steps):** The legend lists five diffusion step configurations:
>     *   Blue circle: 25
>     *   Red square: 50
>     *   Green triangle: 75
>     *   Purple star: 250
>     *   Black diamond: 500
> *   **Data Points:** Multiple data series are plotted, with small numerical annotations (e.g., (0), (2), (16), (32)) placed near various points, likely referencing specific experimental results. The data generally shows a trend where increasing the number of function evaluations leads to a decrease in MMTV, particularly for higher diffusion step counts.
>
> **Panel (b): Posterior inference on Turin, strong priors**
> *   **Title:** (b) Posterior inference on Turin, strong priors.
> *   **Y-axis Range:** MMTV ranges from approximately 0.07 to 0.12.
> *   **Legend (Diffusion Steps):** The legend lists three diffusion step configurations:
>     *   Blue circle: 25
>     *   Red square: 50
>     *   Green triangle: 100
> *   **Data Points:** Similar to Panel (a), data points are plotted and annotated with small numerical labels (e.g., (0), (2), (4), (8)). The overall trend shows that MMTV decreases as the number of function evaluations increases, following the trajectory of the Pareto Frontier.

<center>Figure 3: Pareto frontiers with respect to number of function evaluations (NFEs) and MMTV on posterior inference for OUP and Turin, with varying number of diffusion and Langevin steps. </center>

calls to the score model. In Fig. 3, we visualize the relationship between MMTV, \(N\) and \(N_{L}\) . The Pareto front shows that the best posterior inference is achieved by combining moderate diffusion steps \((N \sim 25 - 50)\) with increasing Langevin corrections if the NFE budget allows it. The sample quality in general improves with more compute, implying that a simple way to calibrate the sampling parameters is to increase NFE until the output distribution does not change.

## 5 DISCUSSION

PriorGuide enables amortized diffusion- based SBI models to adapt to new prior distributions without retraining, an example of the test- time compute paradigm in extending pre- trained model capabilities with dedicated computations at test time which repurpose diffusion guidance for Bayesian inference. In practice, PriorGuide is recommended for moderate- to- high dimensional problems \((4 < D \lesssim 20)\) where simulators are mildly- to- very expensive, making retraining a simulator model burdensome; for settings requiring complex, non- factorized priors, where amortized methods restricted by pre- defined meta- priors (e.g., ACE) yield unsatisfactory performance; and for applications where prior adaptation needs to be fast but not strictly instant—in terms of pure speed, fully amortized methods without test- time compute remain the best choice.

Limitations PriorGuide's effectiveness relies on the new prior \(q(\pmb {\theta})\) having substantial overlap with the training prior \(p_{\mathrm{train}}(\pmb {\theta})\) ; out- of- distribution (OOD) target priors can lead to inaccurate learned scores and unstable guidance. The method also employs approximations—a Gaussian for the reverse transition kernel \(p(\pmb {\theta}_0 | \pmb {\theta}_t, \mathbf{x})\) and a Gaussian mixture model for the prior ratio function \(r(\pmb {\theta})\) —which can introduce inaccuracies, particularly for complex prior ratio shapes. Furthermore, current guidance calculations, involving matrix operations for the GMM components, may pose scalability challenges for high- dimensional parameter spaces \((\dim (\pmb {\theta}) \gg 20)\) . PriorGuide sampling can be computationally intensive: although our method avoids retraining the base model—a key benefit with expensive simulators (e.g., Turin model)—the iterative guided diffusion, particularly with interleaved Langevin refinement steps \((N_{L})\) , incurs a cost. The number of function evaluations (NFEs) increases with diffusion \((N)\) and Langevin steps, creating an accuracy- speed trade- off. This may render PriorGuide less suited than fully amortized methods for applications requiring very rapid inference. Advanced covariance approaches mentioned in Section 3.2 have the potential to speed up the method by requiring fewer NFEs, and represent an interesting future direction.

Conclusions The ability of PriorGuide to decouple expensive simulator runs (for training the base model) from the specification of changing prior beliefs offers significant practical advantages. It allows for post- hoc prior sensitivity analyses and facilitates the direct incorporation of domain expert knowledge post- training, reducing the overall computational footprint in scientific workflows by avoiding the need for repeated model retraining when assumptions change.

---

*Transcribed with OCR and VLMs; text, equations, tables, and figure descriptions may contain mistakes.*
