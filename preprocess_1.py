# rename teams and pitchers
# fill in season

file_name = "train_data.csv"
output_file = "preprocess_1.csv"

team_table = {
    "BPH": 0,
    "DPS": 1,
    "ECN": 2,
    "FBW": 3,
    "GKO": 4,
    "GLO": 5,
    "GUT": 6,
    "HAN": 7,
    "HXK": 8,
    "JBM": 9,
    "JEM": 10,
    "KFH": 11,
    "KJP": 12,
    "MOO": 13,
    "MZG": 14,
    "PDF": 15,
    "PJT": 16,
    "QDH": 17,
    "QPO": 18,
    "RAV": 19,
    "RKN": 20,
    "RLJ": 21,
    "SAJ": 22,
    "STC": 23,
    "UPV": 24,
    "VJV": 25,
    "VQC": 26,
    "XFB": 27,
    "YHA": 28,
    "ZQF": 29,
}
pitcher_table = {
    "abbotan01": 0,
    "abbotco01": 1,
    "adamsau02": 2,
    "adamsch01": 3,
    "adlemti01": 4,
    "adonjo01": 5,
    "aguiaju01": 6,
    "alcanra01": 7,
    "alcansa01": 8,
    "aldegsa01": 9,
    "alexaja01": 10,
    "alexasc02": 11,
    "alexaty01": 12,
    "alexyaj01": 13,
    "allarko01": 14,
    "allenlo02": 15,
    "alvarhe01": 16,
    "alvarjo02": 17,
    "alvarjo03": 18,
    "anderch01": 19,
    "anderco01": 20,
    "anderdr02": 21,
    "anderia01": 22,
    "andersh01": 23,
    "anderta01": 24,
    "andrima01": 25,
    "antonte01": 26,
    "aquinja01": 27,
    "arihako01": 28,
    "armenro01": 29,
    "armstsh01": 30,
    "arrigsp01": 31,
    "ashbyaa01": 32,
    "ashcrgr01": 33,
    "asheral01": 34,
    "assadja01": 35,
    "avilape01": 36,
    "baezmi01": 37,
    "baileho02": 38,
    "bakerbr01": 39,
    "bandaan01": 40,
    "banuema01": 41,
    "bardlu01": 42,
    "barneja01": 43,
    "barnema01": 44,
    "barrija01": 45,
    "bassan01": 46,
    "bassich01": 47,
    "bassobr01": 48,
    "battepe01": 49,
    "bauertr01": 50,
    "baumami01": 51,
    "bautige01": 52,
    "bazsh01": 53,
    "becktr01": 54,
    "beedety01": 55,
    "beeksja02": 56,
    "bellaan01": 57,
    "bellch02": 58,
    "bellobr01": 59,
    "bellova01": 60,
    "bendean01": 61,
    "benjawe01": 62,
    "bergetr01": 63,
    "bergmch01": 64,
    "bernabr01": 65,
    "bettial01": 66,
    "bettich01": 67,
    "biagijo01": 68,
    "bibenau01": 69,
    "bidoos01": 70,
    "bielabr01": 71,
    "birdja01": 72,
    "birdsha01": 73,
    "bivensp01": 74,
    "blachty01": 75,
    "blackma01": 76,
    "blackpa01": 77,
    "blalobr01": 78,
    "blancro01": 79,
    "blazemi01": 80,
    "bleieri01": 81,
    "bolanro01": 82,
    "bolsimi01": 83,
    "bonilli01": 84,
    "borucry01": 85,
    "boshebu01": 86,
    "bowmama01": 87,
    "boydma01": 88,
    "boylejo01": 89,
    "bradlta01": 90,
    "brashma01": 91,
    "brasiry01": 92,
    "braulst01": 93,
    "brebbjo01": 94,
    "bridwpa01": 95,
    "briesbe01": 96,
    "brighje01": 97,
    "brownbe02": 98,
    "brubajt01": 99,
    "bruihju01": 100,
    "buchaja01": 101,
    "buchhcl01": 102,
    "buehlwa01": 103,
    "bumgama01": 104,
    "burkebr01": 105,
    "burneco01": 106,
    "burrobe01": 107,
    "bushky01": 108,
    "bushma01": 109,
    "butleed01": 110,
    "cabreed02": 111,
    "cabrege01": 112,
    "cahiltr01": 113,
    "cainma01": 114,
    "campbpa02": 115,
    "cannigr01": 116,
    "cannojo02": 117,
    "carasma01": 118,
    "carmofa01": 119,
    "carraca01": 120,
    "cashnan01": 121,
    "castada01": 122,
    "castehu01": 123,
    "castery01": 124,
    "castidi01": 125,
    "castilu02": 126,
    "castima03": 127,
    "castrmi01": 128,
    "ceasedy01": 129,
    "ceccosl01": 130,
    "cessalu01": 131,
    "chargjt01": 132,
    "chatwty01": 133,
    "chaveje01": 134,
    "chenwe02": 135,
    "chiriyo01": 136,
    "civalaa01": 137,
    "claseem01": 138,
    "clemepa02": 139,
    "clevimi01": 140,
    "clippty01": 141,
    "cobbal01": 142,
    "colege01": 143,
    "coleta01": 144,
    "colonba01": 145,
    "conlead01": 146,
    "conlopj01": 147,
    "contrro01": 148,
    "corbipa01": 149,
    "cortene01": 150,
    "cottojh01": 151,
    "couloda01": 152,
    "coveydy01": 153,
    "coxau01": 154,
    "cravyty01": 155,
    "crawfku01": 156,
    "crochga01": 157,
    "crousha01": 158,
    "crowewi01": 159,
    "cruzfe01": 160,
    "cuasjo01": 161,
    "cuetojo01": 162,
    "cuevawi01": 163,
    "curryxz01": 164,
    "curtijo02": 165,
    "danaca01": 166,
    "danieda01": 167,
    "danisty01": 168,
    "danksjo01": 169,
    "davieza02": 170,
    "davisau01": 171,
    "davisno01": 172,
    "davisro03": 173,
    "degroja01": 174,
    "dejonch01": 175,
    "delacjo01": 176,
    "delarru01": 177,
    "deleojo03": 178,
    "delgara01": 179,
    "dermoma01": 180,
    "desclan01": 181,
    "detmere01": 182,
    "diazda01": 183,
    "diazjh01": 184,
    "diazmi02": 185,
    "diazyi01": 186,
    "dicker.01": 187,
    "dodddy01": 188,
    "dowdyky01": 189,
    "dubinsh01": 190,
    "duffety01": 191,
    "duffyda01": 192,
    "duggero01": 193,
    "dunnida01": 194,
    "dunnju01": 195,
    "duplajo01": 196,
    "durapmo01": 197,
    "effrosc01": 198,
    "eflinza01": 199,
    "eickhje01": 200,
    "elderbr01": 201,
    "ellisch01": 202,
    "englema01": 203,
    "ennsdi01": 204,
    "eovalna01": 205,
    "erlinro01": 206,
    "eschja01": 207,
    "escobed02": 208,
    "eshelto01": 209,
    "espinpa01": 210,
    "estesjo01": 211,
    "faedoal01": 212,
    "falteba01": 213,
    "fariaja01": 214,
    "farmebu01": 215,
    "fauchca01": 216,
    "feddeer01": 217,
    "feierry01": 218,
    "feldmsc01": 219,
    "felizmi01": 220,
    "feltnry01": 221,
    "ferguca01": 222,
    "festama01": 223,
    "feyerjo01": 224,
    "fiersmi01": 225,
    "fillmhe01": 226,
    "finnebr01": 227,
    "fistedo01": 228,
    "fittsri01": 229,
    "flaheja01": 230,
    "flemijo01": 231,
    "flexech01": 232,
    "flynnbr01": 233,
    "fostema01": 234,
    "francbo01": 235,
    "francjp01": 236,
    "freelky01": 237,
    "friedch01": 238,
    "friedma01": 239,
    "fryja01": 240,
    "fujinsh01": 241,
    "fulmeca01": 242,
    "fulmemi01": 243,
    "funkhky01": 244,
    "gaddihu01": 245,
    "gagnodr01": 246,
    "galleza01": 247,
    "gantjo01": 248,
    "garabge01": 249,
    "garcibr01": 250,
    "garcide01": 251,
    "garcija04": 252,
    "garcilu03": 253,
    "garcilu05": 254,
    "garciri01": 255,
    "garciro03": 256,
    "garream01": 257,
    "garrebr01": 258,
    "garzaju01": 259,
    "garzama01": 260,
    "gassero01": 261,
    "gearrco01": 262,
    "geedi01": 263,
    "germado01": 264,
    "gibsoky01": 265,
    "gilbelo01": 266,
    "gilbety01": 267,
    "gillu01": 268,
    "ginnjt01": 269,
    "givenmy01": 270,
    "glasnty01": 271,
    "godleza01": 272,
    "gonsast01": 273,
    "gonsoto01": 274,
    "gonzach01": 275,
    "gonzagi01": 276,
    "gonzama02": 277,
    "gonzame01": 278,
    "gonzami03": 279,
    "goudeas01": 280,
    "gracego01": 281,
    "gracema02": 282,
    "graveke01": 283,
    "grayjo02": 284,
    "grayso01": 285,
    "greench03": 286,
    "greenco01": 287,
    "greensh02": 288,
    "greinza01": 289,
    "gsellro01": 290,
    "guerrja02": 291,
    "guerrju02": 292,
    "guthrje01": 293,
    "hahnje01": 294,
    "haleda02": 295,
    "hamelco01": 296,
    "hamilia01": 297,
    "hammeja01": 298,
    "hanifbr01": 299,
    "happja01": 300,
    "hardybl01": 301,
    "harrelu01": 302,
    "harriho03": 303,
    "harriky01": 304,
    "harvema01": 305,
    "hatchto01": 306,
    "hauscmi01": 307,
    "heanean01": 308,
    "hearnta01": 309,
    "hellije01": 310,
    "hendrky01": 311,
    "hendrli01": 312,
    "henlebl01": 313,
    "henryto01": 314,
    "hentgsa01": 315,
    "hergeji01": 316,
    "hernaca04": 317,
    "hernada02": 318,
    "hernafe02": 319,
    "hernajo02": 320,
    "herzdj01": 321,
    "hessda01": 322,
    "hestoch01": 323,
    "hicksjo03": 324,
    "hillga02": 325,
    "hillri01": 326,
    "hoeinbr01": 327,
    "hollade01": 328,
    "hollojo01": 329,
    "holmbda01": 330,
    "holmecl01": 331,
    "holmegr01": 332,
    "holtoty01": 333,
    "honeybr01": 334,
    "houckta01": 335,
    "howarsa01": 336,
    "howarsp01": 337,
    "huffda01": 338,
    "hughebr01": 339,
    "hugheph01": 340,
    "hunteto02": 341,
    "hurtky01": 342,
    "hutchdr01": 343,
    "imanash01": 344,
    "iveyty01": 345,
    "iwakuhi01": 346,
    "jacksan01": 347,
    "jacksed01": 348,
    "jacqujo01": 349,
    "jamesdr01": 350,
    "jamesjo02": 351,
    "jarvibr01": 352,
    "javiecr01": 353,
    "jayemy01": 354,
    "jeffeda01": 355,
    "jeffrje01": 356,
    "jimenda01": 357,
    "jimenub01": 358,
    "johnser04": 359,
    "johnsji04": 360,
    "johnsse01": 361,
    "jonesja09": 362,
    "jorgefe01": 363,
    "jungmta01": 364,
    "junkja01": 365,
    "juradar01": 366,
    "kaprija01": 367,
    "kayan01": 368,
    "kellebr01": 369,
    "kellemi03": 370,
    "kelletr01": 371,
    "kellyca01": 372,
    "kellyme01": 373,
    "kellyza01": 374,
    "kendrky01": 375,
    "kennebr02": 376,
    "kenneia01": 377,
    "kerkeor01": 378,
    "kerrra01": 379,
    "kiliaca01": 380,
    "kinghni01": 381,
    "kingmi01": 382,
    "kintzbr01": 383,
    "kirbyge01": 384,
    "kitchau01": 385,
    "kleinph01": 386,
    "klubeco01": 387,
    "knackla01": 388,
    "knebeco01": 389,
    "kochaja01": 390,
    "kochma01": 391,
    "koehlto01": 392,
    "koenija01": 393,
    "kopecmi01": 394,
    "kowarja01": 395,
    "kremede01": 396,
    "lackejo01": 397,
    "lakintr01": 398,
    "lambeji01": 399,
    "lambepe01": 400,
    "lambjo02": 401,
    "lametdi01": 402,
    "latosma01": 403,
    "latzja01": 404,
    "lauerer01": 405,
    "lawde01": 406,
    "lawreca01": 407,
    "leakemi01": 408,
    "leblawa01": 409,
    "leclejo01": 410,
    "leedy01": 411,
    "leeev01": 412,
    "leeza01": 413,
    "leiteja01": 414,
    "leitema02": 415,
    "leonedo01": 416,
    "lestejo01": 417,
    "lewicar01": 418,
    "lewisco01": 419,
    "liberma01": 420,
    "linceti01": 421,
    "liriafr01": 422,
    "litteza01": 423,
    "littllu01": 424,
    "livelbe01": 425,
    "loaisjo01": 426,
    "lockewa01": 427,
    "lodolni01": 428,
    "logueza01": 429,
    "lohseky01": 430,
    "longsa01": 431,
    "lopezja04": 432,
    "lopezjo02": 433,
    "lopezpa01": 434,
    "lopezre01": 435,
    "loupaa01": 436,
    "lowderh01": 437,
    "lowthza01": 438,
    "lucasjo02": 439,
    "lucchjo01": 440,
    "luetglu01": 441,
    "lugose01": 442,
    "luzarje01": 443,
    "lylesjo01": 444,
    "lynchda02": 445,
    "lynnla01": 446,
    "maedake01": 447,
    "mahlety01": 448,
    "maldoan01": 449,
    "manaese01": 450,
    "mannima02": 451,
    "manoaal01": 452,
    "mantijo01": 453,
    "margeni01": 454,
    "marquge01": 455,
    "marshal01": 456,
    "martefr01": 457,
    "martibr01": 458,
    "martica04": 459,
    "martico01": 460,
    "martico02": 461,
    "martida03": 462,
    "martini01": 463,
    "marveja01": 464,
    "matthze01": 465,
    "matusbr01": 466,
    "matzst01": 467,
    "mayermi01": 468,
    "maytr01": 469,
    "mazurad01": 470,
    "mcallza01": 471,
    "mcartja01": 472,
    "mccarbr01": 473,
    "mccarki01": 474,
    "mccauda01": 475,
    "mcclare01": 476,
    "mcclash01": 477,
    "mcderch01": 478,
    "mcfartj01": 479,
    "mcgowky01": 480,
    "mcguide02": 481,
    "mchugco01": 482,
    "mckaybr01": 483,
    "mcraeal01": 484,
    "meansjo01": 485,
    "medinad01": 486,
    "medinlu02": 487,
    "medlekr01": 488,
    "megiltr01": 489,
    "megilty01": 490,
    "meisiry01": 491,
    "mejiahu01": 492,
    "menacr01": 493,
    "mendeyo01": 494,
    "menezco01": 495,
    "mercami01": 496,
    "merriry01": 497,
    "merryju01": 498,
    "meyeral01": 499,
    "meyerma01": 500,
    "middlke01": 501,
    "mikolmi01": 502,
    "mileywa01": 503,
    "millebo06": 504,
    "millebr04": 505,
    "millema03": 506,
    "millesh01": 507,
    "millsal02": 508,
    "milneho01": 509,
    "milonto01": 510,
    "miranar01": 511,
    "mitchbr01": 512,
    "mizeca01": 513,
    "mlodzca01": 514,
    "molinan01": 515,
    "montafr02": 516,
    "monteke01": 517,
    "montera01": 518,
    "montgmi01": 519,
    "moorean02": 520,
    "moorema02": 521,
    "morejad01": 522,
    "moretda01": 523,
    "morgaad01": 524,
    "morgael01": 525,
    "mortoch02": 526,
    "moyaga01": 527,
    "munozro01": 528,
    "murfepe01": 529,
    "musgrjo01": 530,
    "nanceto01": 531,
    "nastrni01": 532,
    "nealza01": 533,
    "neideni01": 534,
    "nelsoji02": 535,
    "nelsoky01": 536,
    "nelsoni01": 537,
    "nelsory01": 538,
    "newcose01": 539,
    "newsolj01": 540,
    "nicasju01": 541,
    "nicolju01": 542,
    "niesejo01": 543,
    "nixja02": 544,
    "nolasri01": 545,
    "norrida01": 546,
    "novaiv01": 547,
    "nunezda01": 548,
    "oakstr01": 549,
    "oberba01": 550,
    "oberhbr01": 551,
    "obrieri01": 552,
    "ollerad01": 553,
    "olsonre01": 554,
    "ortizlu02": 555,
    "ortizlu03": 556,
    "ortka01": 557,
    "osullse01": 558,
    "oswalco01": 559,
    "ottogl01": 560,
    "overtco01": 561,
    "overtdi01": 562,
    "oviedjo01": 563,
    "oviedlu01": 564,
    "owenshe02": 565,
    "paddach01": 566,
    "paganem01": 567,
    "pallaan01": 568,
    "palumjo01": 569,
    "pannoth01": 570,
    "parkebl01": 571,
    "parkemi01": 572,
    "patinlu01": 573,
    "paxtoja01": 574,
    "payamjo01": 575,
    "payanpe01": 576,
    "peacoma01": 577,
    "pelfrmi01": 578,
    "penafe01": 579,
    "penniwa01": 580,
    "pepiory01": 581,
    "peralfr01": 582,
    "peralwi01": 583,
    "perezeu02": 584,
    "perezma02": 585,
    "peterda01": 586,
    "petityu01": 587,
    "pfaadbr01": 588,
    "phelpda01": 589,
    "phillco01": 590,
    "phillev01": 591,
    "phillty01": 592,
    "pilkiko01": 593,
    "pillty01": 594,
    "pinedmi01": 595,
    "pivetni01": 596,
    "plassmi01": 597,
    "plesaza01": 598,
    "plutkad01": 599,
    "pomerdr01": 600,
    "ponceco01": 601,
    "ponceda01": 602,
    "porceri01": 603,
    "poteeco01": 604,
    "povicca01": 605,
    "poynebo01": 606,
    "priceda01": 607,
    "priesqu01": 608,
    "pruitau01": 609,
    "pukaj01": 610,
    "quintjo01": 611,
    "raganco01": 612,
    "ramirer02": 613,
    "ramirjc01": 614,
    "ramirne01": 615,
    "ramirye01": 616,
    "ramosce01": 617,
    "ranauan01": 618,
    "rasmuco02": 619,
    "rasmudr01": 620,
    "rayro02": 621,
    "reaco01": 622,
    "reedco01": 623,
    "reidfse01": 624,
    "reyesal02": 625,
    "reyesde02": 626,
    "richacl01": 627,
    "richaly01": 628,
    "richatr01": 629,
    "roarkta01": 630,
    "roberni01": 631,
    "robleha01": 632,
    "rockeku01": 633,
    "rodgebr01": 634,
    "rodonca01": 635,
    "rodrica02": 636,
    "rodrich01": 637,
    "rodried05": 638,
    "rodriel02": 639,
    "rodrigr01": 640,
    "rodrije01": 641,
    "rodriya01": 642,
    "rodriye01": 643,
    "rogertr01": 644,
    "romansa01": 645,
    "romdr01": 646,
    "romerfe01": 647,
    "rondohe01": 648,
    "rouppla01": 649,
    "rowlech01": 650,
    "rucindr01": 651,
    "rusinch01": 652,
    "rutleja01": 653,
    "ryanjo04": 654,
    "ryanri01": 655,
    "sadleca02": 656,
    "sadzeco01": 657,
    "salech01": 658,
    "samarje01": 659,
    "sampsad01": 660,
    "sampske01": 661,
    "sanchaa01": 662,
    "sanchcr01": 663,
    "sanchsi01": 664,
    "sandlni01": 665,
    "sandopa02": 666,
    "sandsco01": 667,
    "santaer01": 668,
    "santito01": 669,
    "sborzjo01": 670,
    "scherma01": 671,
    "schmicl01": 672,
    "scholje01": 673,
    "schrejo01": 674,
    "schulja02": 675,
    "schwesp01": 676,
    "scottch01": 677,
    "scottta02": 678,
    "scribtr01": 679,
    "seaboco01": 680,
    "searsjp01": 681,
    "selbyco01": 682,
    "senzaan01": 683,
    "severlu01": 684,
    "shawbr01": 685,
    "sheehem01": 686,
    "shephch01": 687,
    "shielja02": 688,
    "shiplbr01": 689,
    "silsech01": 690,
    "simonal01": 691,
    "simslu01": 692,
    "singebr01": 693,
    "skaggty01": 694,
    "skenepa01": 695,
    "skogler01": 696,
    "skubata01": 697,
    "slegeaa01": 698,
    "smeltde01": 699,
    "smithaj01": 700,
    "smithbu03": 701,
    "smithca03": 702,
    "smithch08": 703,
    "smithdr01": 704,
    "smithri01": 705,
    "smylydr01": 706,
    "snellbl01": 707,
    "snideco01": 708,
    "soriajo01": 709,
    "soriajo02": 710,
    "sorokmi01": 711,
    "sotogr01": 712,
    "sparkgl01": 713,
    "spencmi01": 714,
    "spierca01": 715,
    "sprinje01": 716,
    "stammcr01": 717,
    "stanery01": 718,
    "staumjo01": 719,
    "steelju01": 720,
    "stephja01": 721,
    "stewabr01": 722,
    "stockro01": 723,
    "stonega01": 724,
    "stoudle01": 725,
    "strahma01": 726,
    "straida01": 727,
    "strasst01": 728,
    "stratch01": 729,
    "stridsp01": 730,
    "stromma01": 731,
    "suareal01": 732,
    "suarean01": 733,
    "suarejo01": 734,
    "sulseco01": 735,
    "surkaer01": 736,
    "swanser01": 737,
    "swarmma01": 738,
    "syndeno01": 739,
    "szaputh01": 740,
    "taillja01": 741,
    "tanakma01": 742,
    "tarnofr01": 743,
    "tarplst01": 744,
    "taylojo02": 745,
    "teperry01": 746,
    "thompja03": 747,
    "thompke02": 748,
    "thompza02": 749,
    "thorntr01": 750,
    "thorpdr01": 751,
    "thorple01": 752,
    "tillmch01": 753,
    "tinocje01": 754,
    "tomlijo01": 755,
    "toussto01": 756,
    "triggan01": 757,
    "trivilo01": 758,
    "tropeni01": 759,
    "tsengje01": 760,
    "turneja01": 761,
    "tylerky01": 762,
    "ucetaed01": 763,
    "underdu01": 764,
    "urenajo01": 765,
    "valdece01": 766,
    "vargace01": 767,
    "vargaja01": 768,
    "varlalo01": 769,
    "vasqura02": 770,
    "velasvi01": 771,
    "velazhe01": 772,
    "ventejo01": 773,
    "ventuyo01": 774,
    "verlaju01": 775,
    "verrelo01": 776,
    "vesiaal01": 777,
    "vinceni01": 778,
    "vinesda01": 779,
    "vogelry01": 780,
    "volstch01": 781,
    "vothau01": 782,
    "wachami01": 783,
    "wagueja01": 784,
    "wainwad01": 785,
    "waldike01": 786,
    "waldrhu01": 787,
    "walkery01": 788,
    "walketa01": 789,
    "walstbl01": 790,
    "wantzan01": 791,
    "warread01": 792,
    "warrewi01": 793,
    "watkisp01": 794,
    "weathry01": 795,
    "webblo01": 796,
    "weberry01": 797,
    "wellsty01": 798,
    "wentzjo01": 799,
    "whalero01": 800,
    "whitebr01": 801,
    "whitemi03": 802,
    "whitlch01": 803,
    "whitlga01": 804,
    "wicksjo01": 805,
    "widenta01": 806,
    "wielajo01": 807,
    "wilkad01": 808,
    "willibr02": 809,
    "williga01": 810,
    "willitr01": 811,
    "wilsobr02": 812,
    "wilsost02": 813,
    "wilsoty01": 814,
    "winanal01": 815,
    "wingetr01": 816,
    "winnke01": 817,
    "wislema01": 818,
    "wittgni01": 819,
    "wojcias01": 820,
    "wolfja01": 821,
    "woobr01": 822,
    "woodal02": 823,
    "woodfja01": 824,
    "woodhu01": 825,
    "woodssi01": 826,
    "woodtr01": 827,
    "worleva01": 828,
    "wrighda04": 829,
    "wrighky01": 830,
    "wrighmi01": 831,
    "wrighst01": 832,
    "wroblju01": 833,
    "yajurmi01": 834,
    "yamamjo01": 835,
    "yamamyo01": 836,
    "yarbrry01": 837,
    "ynoaga01": 838,
    "ynoahu01": 839,
    "youngal01": 840,
    "zerpaan01": 841,
    "zeuchtj01": 842,
    "zimmejo02": 843,
}

weight_table = [
    [
        1.000000,
        0.882497,
        0.606531,
        0.324652,
        0.135335,
        0.043937,
        0.011109,
        0.002187,
        0.000335,
    ],
    [
        0.882497,
        1.000000,
        0.882497,
        0.606531,
        0.324652,
        0.135335,
        0.043937,
        0.011109,
        0.002187,
    ],
    [
        0.606531,
        0.882497,
        1.000000,
        0.882497,
        0.606531,
        0.324652,
        0.135335,
        0.043937,
        0.011109,
    ],
    [
        0.324652,
        0.606531,
        0.882497,
        1.000000,
        0.882497,
        0.606531,
        0.324652,
        0.135335,
        0.043937,
    ],
    [
        0.135335,
        0.324652,
        0.606531,
        0.882497,
        1.000000,
        0.882497,
        0.606531,
        0.324652,
        0.135335,
    ],
    [
        0.043937,
        0.135335,
        0.324652,
        0.606531,
        0.882497,
        1.000000,
        0.882497,
        0.606531,
        0.324652,
    ],
    [
        0.011109,
        0.043937,
        0.135335,
        0.324652,
        0.606531,
        0.882497,
        1.000000,
        0.882497,
        0.606531,
    ],
    [
        0.002187,
        0.011109,
        0.043937,
        0.135335,
        0.324652,
        0.606531,
        0.882497,
        1.000000,
        0.882497,
    ],
    [
        0.000335,
        0.002187,
        0.011109,
        0.043937,
        0.135335,
        0.324652,
        0.606531,
        0.882497,
        1.000000,
    ],
]


# pitcher = []

# with open(file_name, 'r') as file:
#     id1 = 0
#     file.readline()
#     lines = file.readlines()
#     for l in lines:
#         l = l.strip().split(',')
#         pitcher.append(l[6])
#         pitcher.append(l[7])

# pit = np.unique(np.array(pitcher))
# np.sort(pit)
# i = 0
# for p in pit[1:]:
#     print(f'\'{p}\': {i}, ', end='')
#     i += 1

with open(output_file, "w") as o_f:
    with open(file_name, "r") as file:
        id1 = 0
        title = file.readline()
        o_f.writelines(title)
        lines = file.readlines()
        for l in lines:
            l = l.strip().split(",")
            # team_name to id
            l[1] = team_table[l[1]]
            l[2] = team_table[l[2]]
            # pitcher name to id
            if l[6] != "":
                l[6] = pitcher_table[l[6]]
            if l[7] != "":
                l[7] = pitcher_table[l[7]]
            # fill the year
            if l[12] == "":
                l[12] = l[3][:4]

            newl = []
            for i in range(len(l)):
                newl.append(str(l[i]))
                newl.append(",")
            newl.pop()
            newl.append("\n")
            o_f.writelines(newl)

        # for line in file:
        #     parts = line.strip().split(',')
        #     for i in len(parts):


# print(tim)
# team = np.unique(np.array(tim))
# np.sort(team)
