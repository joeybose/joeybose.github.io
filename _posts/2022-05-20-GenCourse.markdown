---
layout: post
title:  "COMP760: Geometry and Generative Models"
date:   2022-05-20 10:08:27 -0500
permalink: /Blog/GenCourse
comments: False
share: True
categories: Course
---
# Co-Instructors:  [Joey Bose](https://joeybose.github.io/) and [Prakash Panangaden](https://www.cs.mcgill.ca/~prakash/)

# Course Overview
<img src="../assets/my_assets/HyperFlow_15min/hyperflow_animation_large.gif" alt="" style="border-radius:0%;height: 30%;width: 70%;margin-left: auto;margin-right: auto;position: relative;display: block; padding-top: 10px; padding-bottom: 10px">

<!--<span style="font-family:Helvetica; font-size:1.25em;">-->
In recent years Deep Generative Models have seen remarkable success over a variety of data domains such as images, text, and audio to name a few. However, the predominant approach in many of these models (e.g. GANS, VAE, Normalizing Flows) is to treat data as fixed-dimensional continuous vectors in some Euclidean space, despite significant evidence to the contrary (e.g. 3D molecules). This course places a direct emphasis on learning generative models for complex geometries described via manifolds, such as spheres, tori, hyperbolic spaces, implicit surfaces, and homogeneous spaces. The purpose of this seminar course is to understand the key design principles that underpin the new wave of geometry-aware generative models that treat the rich geometric structure in data as a first-class citizen. This seminar course will also serve to develop extensions to these approaches at the leading edge of research and as a result, a major component of the course will focus on class participation through presenting papers and a thematically-relevant course project.
<!--</span>-->

* Location: Mila Auditorium 1
* Time: Fridays 1pm - 4pm
* Office Hours: By email correspondance.
* [Anonymous Feedback Form](https://forms.gle/g4xns6vizBsSKCq39)

# Prerequisites
This course is designed to bring students to the current frontier of knowledge on geometric generative models so that ideally, their course projects can make a novel contribution that can either be algorithmic, theoretical, or empirical. A previous background in machine learning is strongly recommended. Linear algebra, basic multivariate calculus, basics of working with probability, and programming skills are required. No background in geometry or generative models is needed for the course but any such knowledge may aid in a deeper engagement with the course material. However, it is strongly recommended that this course is not the first—or even second—Machine Learning course, and if there are any specific doubts please contact the instructor for special permission or equivalency of prerequisites.

* Math 223 (Linear Algebra 2) or equivalent
* COMP 551 (Applied Machine Learning) or equivalent
* Math 323 (Probability) or equivalent
* Math 222 (Calculus 3) or equivalent


# Course Structure
The first third of the course will consist of lectures covering necessary background material in geometric deep learning and generative models, after which every week will focus on student presentations on a pre-selected topic, using a couple of papers as reference. Each student must prepare one presentation individually or as a pair for the semester which is independent of their course project. The final two weeks will be reserved for course project presentations.
In-class discussion will center around:

* Understanding the strengths and weaknesses of these methods.
* Understanding the relationships between these methods, and with previous approaches.
* Extensions or applications of these methods.
* Experiments that might better illuminate their properties.

# Grading
There will be an ungraded but mandatory project extended abstract/project proposal submission on Week 6 of the course.
The course project, including project proposal, must be formatted in Latex using the NeurIPS 2022 (Preprint
option) template. [Latex Template](https://neurips.cc/Conferences/2022/CallForPapers)

* 20% In-class paper presentations
* 30% Student project presentation
* 50% Student project report

# Course Project
The course project can be done individually, in pairs, or in rare cases teams
of 3. The project will be evaluated wholistically, on a number of
criterias---not unsimilar to an actual review of a conference paper. The grading will take into account the number of people on a project.
Some of these include: novelty of the idea, how well you present them in the report, how clearly you position your work relative to existing literature, how illuminating your experiments are, and well-supported your conclusions are.


Each course project will require a 2 page project proposal which can be thought of as an extended abstract. The actual project report will need to be 4-8 pages in length in the format of a NeurIPS paper in Preprint mode [Latex template](https://neurips.cc/Conferences/2022/PaperInformation/StyleFiles). The report should include
an abstract, introduction, coverage of related work, a minimum viable project with some demonstration either theoretical or empirical of the ideas soundness, and if needed some future looking thoughts on how to proceed if given more time.



# Reading List:
## Textbooks
There is no required textbook for this course but the ["Geometric Deep Learning proto book"](https://geometricdeeplearning.com/) is a good reference for much of the technical foundations covered in this course. Another fantastic resource is the ["Differential Geometry for Generative Modeling"](http://www2.compute.dtu.dk/~sohau/weekendwithbernie/Differential_geometry_for_generative_modeling.pdf) textbook.


## Prerequisite readings/material:
* [Self-Assessment Quiz](https://github.com/joeybose/comp760_lecturenotes/blob/master/COMP760___Self_Assessment_Quiz.pdf)
* [Automatic Differentiation and Variational Inference](http://arxiv.org/pdf/1603.00788v1.pdf)
* [Normalizing Flows for Probabilistic Modeling and Inference](https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf)

## Sept 2nd - Week 1: Geometry Primer Part I (Prakash Lectures):
* [Lecture Notes](https://github.com/joeybose/comp760_lecturenotes/blob/master/Week_1__Geometry_Primer_Part_I.pdf)
* [Week 1 Administrative Slides](https://github.com/joeybose/comp760_lecturenotes/blob/master/COMP760%20Week%201_slides.pdf)
* Topology: open and closed sets, continuous functions, convergence, metrics, compactness.
* Manifolds I
* Smoothness
* Charts

## Sept 9th - Week 2: Geometry Primer Part II (Prakash Lectures):
* [Lecture Notes](https://github.com/joeybose/comp760_lecturenotes/blob/master/Week_2__Geometry_Primer_Part_II.pdf)
* Manifolds II
* Tangent Vectors
* Tensors
* Tangent Bundle
* Metrics
* Affine connections
* Lie groups

## Sept 16th - Week 3: Deep Generative Models Primer Part I (Joey Lectures)
Deep generative models learn to transform unstructured noise to highly structured data like natural images. While these models may come in various forms but they can be broadly classified as either likelihood-based or implicit models. The former model class is already quite rich with popular modeling families such as VAE’s, and Normalizing Flows and will be the starting point for this week's topics.

* [Lecture Notes](https://github.com/joeybose/comp760_lecturenotes/blob/master/Week_3__Generative_Models_I.pdf)
* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
* [Variational Inference with Normalizing Flows](http://proceedings.mlr.press/v37/rezende15.pdf)
* [Density Estimation with RealNVP](https://arxiv.org/abs/1605.08803)
* [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

## Sept 23rd - Week 4: Deep Generative Models Primer Part II (Joey Lectures)
In recent years there has been a resurgence of old ideas but repackaged for a modern time. Specifically, score matching and diffusion models have existed in some form or other prior to their recent renaissance but the main technical novelties lie—beyond new techniques that allow for fast training, inference, and impressive sampling quality—in their connection to Stochastic Differential Equations, VAE’s and Continuous Normalizing Flows. This week will focus on exploring these connections in detail and as we will see many new ideas are old ideas in disguise.

* [Lecture Notes](https://github.com/joeybose/comp760_lecturenotes/blob/master/Week_4__Generative_Models_II.pdf)
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)
* [Denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239)
* [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
* [Score Matching Blog Post](https://yang-song.github.io/blog/2021/score/)

## Sept 30th - Week 5: Distributions on Manifolds
Defining probability distributions on manifolds requires specific care as many familiar notions in Euclidean geometry become incompatible for general manifolds. For example, the classical Gaussian distribution has at least three different instantiations in the manifold setting, e.g. Restricted, Wrapped, and the Riemannian normal distribution. We will focus on covering mainly the Riemannian normal distribution as well as the Von Mises Fisher distribution which is typically used in Spherical geometry.

### Core Readings
* [Lecture Notes](https://github.com/joeybose/comp760_lecturenotes/blob/master/Week_5__Distributions_on_Manifolds.pdf)
* [Reparameterizing Distributions on Lie Groups](https://arxiv.org/abs/1903.02958)
* [Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements](https://hal.inria.fr/inria-00614994/PDF/Pennec.JMIV06.pdf)
* [Dispersion on a Sphere](http://palaeo.spb.ru/pmlibrary/pmpapers/fisher_1953.pdf)


### Extra Readings
* [Directional data analysis under the general projected normal distribution](https://www.sciencedirect.com/science/article/pii/S1572312712000457)
* [Riemannian Gaussian Distributions on the Space of Symmetric Positive Definite Matrices](https://arxiv.org/abs/1507.01760)

## Oct 7th - Week 6: Spherical Geometry
**Project Proposals Due**

The first wave of geometry-aware deep generative models focused largely on spherical geometry—i.e. Riemannian manifolds with positive curvature—as it is perhaps most accessible after Euclidean geometry. As a result, there are a few prominent work that exploit this structure to define hyperspherical latent spaces, or more complex distributions on the Sphere using Normalizing Flows or even GANs.

### Core Readings
* [Lecture Notes](https://github.com/joeybose/comp760_lecturenotes/blob/master/Week_6__Spherical_Geometry.pdf)
* [Hyperspherical Variational Auto-Encoders](https://arxiv.org/abs/1804.00891)
* [Normalizing Flows on Tori and Spheres](https://arxiv.org/abs/2002.02428)
* [The Power Spherical distribution](https://arxiv.org/abs/2006.04437)

### Extra Readings
* [Variational Autoencoders with Riemannian Brownian Motion Priors](https://arxiv.org/pdf/2002.05227)
* [Sphere Generative Adversarial Network Based on Geometric Moment Matching](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Sphere_Generative_Adversarial_Network_Based_on_Geometric_Moment_Matching_CVPR_2019_paper.pdf)

## Oct 14th - Week 7: Hyperbolic Geometry
Hyperbolic spaces—i.e. manifolds with constant negative curvature—have become an increasingly useful geometry in the modern machine learning toolkit. From modelling social networks, trees, biological networks, to hierarchical diffusion processes hyperbolic spaces have found tremendous practical advantages over Euclidean counterparts. Naturally, extending generative models to hyperbolic space requires taking into account manifold specific operations like the exponential and logarithmic maps, parallel transport all of which are key design decisions when constructing neural architectures that operate on these spaces.

### Core Readings
* [Lecture Notes](https://github.com/joeybose/comp760_lecturenotes/blob/master/Week_7__Hyperbolic_Geometry.pdf)
* [Google Collab Wrapped Normal Tutorial](https://colab.research.google.com/drive/1kSdmi2r6QMO7gI8YPqeiWoW2ayxcu1MY)
* [Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders](https://arxiv.org/abs/1901.06033)
* [A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning](https://arxiv.org/abs/1902.02992)
* [Latent Variable Modelling with Hyperbolic Normalizing Flows](https://arxiv.org/abs/2002.06336)

## Oct 21st - Week 8: Product and Latent Manifolds
Disentanglement is perhaps one of the key goals of unsupervised learning. For generative models, this effectively means isolating the main generative factors that give rise to observed data. Modeling such generative factors as latent variables in VAE like the setup is the dominant paradigm to infuse geometric priors about the data into practical modeling inductive biases. This week we will turn our attention to the various types of geometric structure that one can attach to a latent space such as group structure and product manifolds.

### Core Readings
* [Towards a definition of disentangled representations](https://arxiv.org/abs/1812.02230)
* [Learning mixed-curvature representations in product spaces](https://openreview.net/forum?id=HJxeWnCcF7)
* [Mixed-curvature Variational Autoencoders](https://arxiv.org/abs/1911.08411)

### Extra Readings
* [Latent Space Oddity: on the Curvature of Deep Generative Models](https://arxiv.org/abs/1710.11379)
* [Metrics for deep generative models](https://arxiv.org/abs/1711.01204)
* [Variational Autoencoders with Riemannian Brownian Motion Priors](https://arxiv.org/abs/2002.05227)
* [Pulling back information geometry](https://arxiv.org/abs/2106.05367)


## Oct 28th - Week 9: Normalizing Flows on Riemannian Manifolds
How can we define flexible probability distributions on general Riemannian manifolds? Turns out one natural way to do so is to define an easy-to-sample prior distribution and a time-evolving vector field that transports this density to the desired target. This effectively generalizes the continuous normalizing flow approach previously seen in week 3 to Riemannian manifolds and this week we will cover 3 papers that were published concurrently on this very topic.
Extending the CNF’s to manifolds typically requires backpropping through an ODE solver which is computationally expensive. In this week we will cover a series of normalizing flows that sidestep this expensive computation by using various methodological innovations such as using convex potentials from Riemannian optimal transport to neural implementation of Moser’s trick which led to MoserFlow (NeurIPS 2021 outstanding paper).

### Core Readings
* [Riemannian Continuous Normalizing Flows](https://arxiv.org/abs/2006.10605)
* [Moser Flow: Divergence-based Generative Modeling on Manifolds](https://arxiv.org/abs/2108.08052)
* [Matching Normalizing Flows and Probability Paths on Manifolds](https://arxiv.org/pdf/2207.04711)

### Extra Readings
* [Neural Manifold Ordinary Differential Equations](https://arxiv.org/abs/2006.10254)
* [Neural Ordinary Differential Equations on Manifolds](https://arxiv.org/abs/2006.06663)
* [The Riemannian Geometry of Deep Generative Models](https://ieeexplore.ieee.org/document/8575533)
* [Riemannian Convex Potential Maps](https://arxiv.org/abs/2106.10272)

## Nov 4th - Week 10: Equivariant Generative Models
Much of observed data is a result of physical processes which have symmetries. These symmetries manifest themselves as equivariances and invariances to certain transformation groups, e.g. translation, rotation, scaling, etc …, and imbuing generative models with these structural inductive biases is a core design principle. In this week we will cover many types of equivariant generative models and their application to physics, molecular dynamics, and many more practical domains.

### Core Readings
* [Equivariant Flows: exact likelihood generative learning for symmetric densities](https://arxiv.org/abs/2006.02425)
* [Equivariant Manifold Flows](https://arxiv.org/abs/2107.08596)
* [Group Equivariant Generative Adversarial Networks](https://arxiv.org/pdf/2005.01683)
* [E(n) Equivariant Normalizing Flows](https://arxiv.org/abs/2105.09016)

### Extra Readings
* [Equivariant Finite Normalizing Flows](https://arxiv.org/abs/2110.08649)
* [Implicit Riemannian Concave Potential Maps](https://arxiv.org/abs/2110.01288)
* [Sampling SU(N) with gauge equivariant flows](https://arxiv.org/abs/2008.05456)

## Nov 11th - Week 11: Geometric Score and Diffusion Models
Score and Diffusion models are the current state of the art generative models for both likelihood estimation as well sample quality. Given their performance for images
it is a natural question on whether these models can be adapted to more complex geometries. A recent influx of papers definitively answer this question with compelling samples ranging from spherical, hyperbolic and toroidal geometry but also practical use cases in molecular simulations.

### Core Readings
* [Equivariant Diffusion for Molecule Generation in 3D](https://arxiv.org/abs/2203.17003)
* [Riemannian Score-Based Generative Modeling](https://arxiv.org/abs/2202.02763)
* [Riemannian Diffusion Models](https://arxiv.org/pdf/2208.07949)

## Nov 18th - Week 12: Student Presentations Part I

## Nov 25th - Week 13: Student Presentations Part II
** Last class of the semester. Projects reports due 1 week after this class on
Dec 2nd**

{% if page.comments %}

<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://joeybose.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

{% endif %}
