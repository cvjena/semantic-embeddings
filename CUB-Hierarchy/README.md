Class Hierarchy for CUB-200-2011
================================

This directory contains hierarchical information about the 200 bird classes in the [Caltech-UCSD Birds-200-2011][1] dataset.

[**classes_wikispecies.txt**](classes_wikispecies.txt) maps the numerical class labels (ranging from 1 to 200) to the scientific names of the birds.
For determining these, we searched [Wikispecies][2] for the respective English name of the bird and used [Wikidata][3] as a fall-back if the bird could not be found in Wikispecies.

[**hierarchy_wikispecies.txt**](hierarchy_wikispecies.txt) defines the class taxonomy in a human-readable tree format.
This taxonomy corresponds to the one provided by [Wikispecies][2], where we used the following taxonomy levels:

- ***ordo*** (ending on *-formes*)
- ***subordo***
- ***superfamilia*** (ending on *-oidea*)
- ***familia*** (ending on *-idae*)
- ***subfamilia*** (ending on *-inae*)
- ***genus***
- ***species*** (consisting of two words, the first one being the *genus*)

Note that *subordo*, *superfamilia*, and *subfamilia* only exist in some branches of the hierarchy and are denoted by comments in parentheses.
The root node of the hierarchy is the class *Aves* (birds).

[**hierarchy_balanced.txt**](hierarchy_balanced.txt) is a derivation of the Wikispecies-based taxonomy, where we added some *subordines*, *superfimiliae*, *subfamiliae*, and *tribes*, so that all species have the same depth in the resulting hierarchy.  
We grounded this extension on systematic information found in the English [Wikipedia][4] or, as a fall-back, the [Open Tree of Life][5].
In some cases, where we could not find sufficient information, we had to make up some intermediate levels. These are indicated by question marks following their names.

Moreover, we have introduced an additional first level dividing the *ordines* into 5 groups of *superordines* and *clades* based on information from [Wikipedia][4]:

- *Aequorlitornithes* (water birds)
- *Telluraves* (land birds)
- *Cypselomorphae* (nightjars, nighthawks, swifts, hummingbirds etc.)
- *Columbaves* (cuckoos, turacos, bustards, pigeons, mesites, sandgrouses)
- *Galloanserae* (fowl)

[**hierarchy_flat.txt**](hierarchy_flat.txt), on the other hand, is derived from the Wikispecies-based taxonomy by removing all *subordo*, *superfamilia*, and *subfamilia* levels, hence resulting in a very flat but balanced hierarchy comprising only the levels *ordo*, *familia*, *genus*, and *species*.

The script [**encode_hierarchy.py**](encode_hierarchy.py) can be used to translate these human-readable taxonomies into machine-readable pairs of parent-child tuples, where each node is encoded with a numeric class label.
The remaining files in this directory are the results of this process for each of the three hierarchies.


Known Issues
------------

- Class 130 (**Tree Sparrow**) mixed images of two classes from different *familiae*:
  There are 34 images of *Spizelloides arborea* (American Tree Sparrow) and 26 images of *Passer montanus* (Tree Sparrow).
  Though the ratio is quite balanced, we mapped this class to the species with slightly more images, i.e., *Spizelloides arborea*.
  Ideally, one should split this class into two separate ones, but we wanted to maintain the original class structure of CUB-200-2011 for comparability.

- Class 91 (**Mockingbird**) is actually an informal group of species from 3 *genera* of the *familia* *mimidae*, but the images seem to mainly show instances of the *species* *Mimus polyglottos*, which is the only mockingbird commonly found North America. Thus, we map that class to this *species*.

- While most classes have a species-level resolution, a handful of them are coarser:
    - Class 44 (**Frigatebird**) resolves to *Fregata* (*genus* level), which encompasses 5 bird species of quite different appearance.
    - Class 92 (**Nighthawk**) resolves to *Chordeiles* (*genus* level), which comprises 6 species.
    - Class 103 (***Sayornis***) is at the *genus* level, comprising 3 species.
    - Class 110 (***Geococcyx***) is at the *genus* level, but there are only 2 possible species with minor visual differences.


Citation
--------

If you use this hierarchy for the CUB dataset in your work, please cite the following paper:

> [**Deep Learning on Small Datasets without Pre-Training usine Cosine Loss.**][6]  
> Björn Barz and Joachim Denzler.  
> IEEE Winter Conference on Applications of Computer Vision (WACV), 2020.


[1]: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
[2]: https://species.wikimedia.org/
[3]: https://www.wikidata.org/
[4]: https://en.wikipedia.org/
[5]: https://tree.opentreeoflife.org/
[6]: https://arxiv.org/pdf/1901.09054