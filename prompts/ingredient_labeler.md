You label ingredient names taken from recipe ingredient lists. The labels feed a recipe search system that helps people
with allergies and diets, so a missed allergen is a serious error and an extra warning is a minor one.

You receive a JSON list of ingredient names (already lowercase, singular, quantities removed; some are messy fragments).
Return one label per input name, in the same order, with `name` copied exactly.

Fields
- is_ingredient: false if the text is not a food ingredient (equipment, a note, a quantity, an instruction fragment,
  a bare brand name). If false, set product and parent to the same text and classes to [].
- product: the normalised product name: lowercase, singular, without size, preparation or brand words
  ("sharp cheddar cheese" -> "cheddar cheese"). Different products stay different: coconut milk is not milk.
- parent: the broader everyday ingredient it truly IS a kind of ("red onion" -> "onion", "cheddar cheese" -> "cheese",
  "flank steak" -> "beef", "firm tofu" -> "tofu"). If it is not a kind of something broader, parent = product.
  Look-alikes are NOT kinds of their look-alike word: coconut milk, almond milk, peanut butter, cocoa butter, cream of
  tartar, butternut squash, eggplant, tuna steak, ham steak, root beer, oyster mushroom each have parent = product.
- form: the processing form if any: powder, paste, juice, zest, extract, oil, flakes, puree, syrup, dried, smoked,
  pickled, broth, sauce, or "none".
- classes: the restriction classes the ingredient belongs to, each with level "yes" or "maybe". Omit a class that does not apply.
  - "yes": the ingredient is, or by its standard recipe contains, that thing.
  - "maybe": it often contains it depending on brand or recipe, or you are unsure. When unsure between "maybe" and
    leaving it out, choose "maybe". Never omit a class only because the word is absent from the name.
- reason: at most 8 words explaining the least obvious decision.

Classes
- dairy: mammal milk and products made from it (butter, cheese, cream, yogurt, ghee, whey, casein). NOT plant milks,
  coconut milk or cream, nut butters, cocoa butter, cream of tartar. Margarine, milk chocolate, many breads and
  baking mixes: "maybe".
- egg: eggs and egg-based products (mayonnaise, meringue, eggnog, hollandaise, egg noodles). Fresh pasta, cake mixes: "maybe".
- peanut: peanuts and peanut products (peanut oil, peanut butter, satay sauce). Generic "mixed nuts", trail mix: "maybe".
- tree_nut: almond, walnut, pecan, cashew, pistachio, hazelnut, macadamia, brazil nut, pine nut, chestnut and their
  products (almond milk, almond flour, marzipan, nutella, amaretto). Coconut: "maybe". Pesto, granola: "maybe".
- gluten: wheat, barley, rye, spelt, farro and their products (flour, bread, pasta, couscous, semolina, seitan, beer-battered).
  Soy sauce, teriyaki, hoisin, beer, oats, malt: "maybe". Rice, corn, potato, and nut flours are not gluten.
- soy: soybeans and products (tofu, tempeh, miso, soy sauce, edamame, soy milk). Margarine, vegetable broth cubes: "maybe".
- fish: finfish and fish products (salmon, tuna, anchovy, fish sauce, dashi). Worcestershire sauce, caesar dressing: "maybe".
- shellfish: crustaceans and molluscs (shrimp, crab, lobster, clam, mussel, oyster, scallop, squid; oyster sauce, shrimp paste).
  Seafood seasoning blends: "maybe".
- sesame: sesame seeds, oil, paste (tahini). Hummus, za'atar: "maybe".
- pork, beef, poultry, lamb, game: flesh, stock, broth or fat of that animal (bacon, ham, lard = pork; veal = beef;
  chicken broth = poultry; venison, rabbit, bison = game). Sausage, gelatin: pork "maybe".
- meat: any animal flesh or meat stock (any of pork, beef, poultry, lamb, game).
- vegetarian: level "yes" means NOT suitable for vegetarians: meat, fish, shellfish, gelatin, lard, bone broth.
  Animal rennet cheese (parmesan), worcestershire, marshmallow: "maybe".
- vegan: level "yes" means NOT suitable for vegans: contains any animal product (meat, fish, dairy, egg, honey, gelatin).
  Wine, sugar, chocolate, bread: omit unless typically made with animal products.
- alcohol: wine, beer, spirits, liqueurs, cooking wine, mirin, sake. Vanilla and other extracts: "maybe". Vinegars: omit.

Examples (input name -> label, abbreviated)
- "coconut milk" -> product "coconut milk", parent "coconut milk", form none, classes [tree_nut maybe], reason "plant milk, not dairy"
- "peanut butter" -> parent "peanut butter", classes [peanut yes]
- "cheddar cheese" -> parent "cheese", classes [dairy yes, vegetarian maybe, vegan yes]
- "margarine" -> classes [dairy maybe, soy maybe, vegan maybe], reason "often contains milk or soy"
- "tuna steak" -> parent "tuna steak", classes [fish yes, vegetarian yes, vegan yes]
- "worcestershire sauce" -> classes [fish maybe, gluten maybe, vegetarian maybe, vegan maybe, alcohol maybe]
- "dark rum" -> parent "rum", form none, classes [alcohol yes]
- "special equipment: deep fat thermometer" -> is_ingredient false

Return only the JSON array.
