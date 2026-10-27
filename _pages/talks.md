---
layout: post
title: "Talks"
permalink: /talks/
---

## [Learning Generative Maps and the Rewards That Guide Them](/talks/learning-generative-maps/)

Invited talk · October 2026

[Slides](/talks/learning-generative-maps/) · [Normalizing Flow Maps paper](https://arxiv.org/abs/2610.22711) · [Normalizing Flow Maps blog post](https://the-bose-lab.github.io/normalizing-flow-maps/) · [MARTINI paper](https://arxiv.org/abs/2610.25502) · [MARTINI project page](https://alecsargood.github.io/martini/)

Learning a generative model gives us a sampler for a data distribution. Adapting that sampler to a downstream task requires rewarding preferential outcomes typically in the form of scalar rewards or rankings that the model must internalize. In this talk, I will discuss learning generative maps as well as new methodology to exploit (multiple) rewards at inference used to guide them. I will first introduce Normalizing Flow Maps (NFM), a new generative modeling class which combines few-step generation with exact likelihood evaluation by learning invertible transport maps along a generative probability path. NFMs are centered around two matching principles: pointwise and distributional matching provide complementary training objectives, while access to sample densities enables direct likelihood-based post-training.

In the second part of the talk, I will focus on application domains where the objective of interest requires costly human judgments or experimental measurements such that the true oracle reward is unavailable at inference. However, aiding us in these settings are often cheaper reward surrogates that capture only parts of the desired oracle reward. I will introduce MARTINI, which uses limited oracle scores or rankings to learn a composition of these surrogates. The resulting reward can guide generation without further oracle queries. We characterize what information about the oracle can be recovered from the surrogates and, in the small-tilt regime, relate reward approximation error to the discrepancy between the desired and induced distributions. Examples from image generation and post-training, language steering, and molecular modeling will illustrate how the structure of the generative map and the information available about the reward determine the adaptations we can achieve.

*The slides run in your browser: click or use the arrow keys to advance, and press f for full screen.*
