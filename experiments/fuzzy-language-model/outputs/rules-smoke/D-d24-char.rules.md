# Rulebook: `D-d24-char.pt`

config: `{'vocab_size': 98, 'd_model': 24, 'n_layers': 2, 'n_heads': 8, 'mixer': 'fuzzydelta', 'ffn': 'tsk', 'ffn_mult': 1.5, 'tsk_dim': 16, 'n_rules': 16, 'decay': 'data', 'shortconv': 8, 'exp_norm': 'mean', 'ffn_exp_norm': 'mean', 'width_share': 'full', 'n_unique': 0, 'ffn_order': 1, 'ffn_width': '', 'ffn_norm': 'none', 'ante_taps': 1, 'max_len': 256}`
tokenizer: `character`

## Layer 0 TSK feed-forward (first-order, raw firing): 36 rules, 34 win > 0.1% of positions

- **rule 19**: wins 14.0%, mean firing 0.207
  - fires on contexts ending in `v`, `"`, `,`; strongest: `want·the·box,"` · `hare·the·box,"` · `ty·learned·a·v` · `a·v` · `they·saw·the·v`
  - gain where it wins: +3.10 ± 1.48; THEN raise `B` 0.44, `S` 0.12, `L` 0.10, `T` 0.06, `H` 0.03, `q` 0.03; when the gain is negative, raise `·` 0.72, `⏎⏎` 0.11, `'` 0.07, `.` 0.02, `k` 0.02, `!` 0.01
- **rule 30**: wins 11.2%, mean firing 0.244
  - fires on contexts ending in `o`, `j`, `r`; strongest: `.·The·bird·scr` · `g·tree.↵The·mo` · `s·ball.↵The·mo` · `ome·was·the·mo` · `ot·find·the·mo`
  - gain where it wins: -1.17 ± 1.33; THEN raise `m` 0.16, `·` 0.14, `f` 0.08, `n` 0.07, `s` 0.07, `,` 0.06; when the gain is negative, raise `I` 0.25, `a` 0.17, `e` 0.15, `A` 0.12, `E` 0.07, `i` 0.05
- **rule 24**: wins 8.9%, mean firing 0.217
  - fires on contexts ending in `o`, `u`, `i`; strongest: `y·one·box,"` · `hare·the·box,"` · `want·the·box,"` · `·the·knot!"↵Ju` · `·you,·frog!·Yo`
  - gain where it wins: +2.83 ± 0.51; THEN raise `A` 0.32, `S` 0.19, `H` 0.14, `I` 0.13, `T` 0.04, `M` 0.03; when the gain is negative, raise `.` 0.45, `·` 0.22, `?` 0.14, `!` 0.12, `,` 0.04, `l` 0.01
- **rule 21**: wins 8.1%, mean firing 0.253
  - fires on contexts ending in `o`, `u`, `i`; strongest: `.·They·all·enj` · `anted·to·rescu` · `·her·face.↵Whi` · `·to·find·the·o` · `m·tasted·the·o`
  - gain where it wins: +2.14 ± 1.18; THEN raise `r` 0.70, `n` 0.23, `m` 0.02, `l` 0.02, `g` 0.01, `t` 0.01; when the gain is negative, raise `A` 0.55, `"` 0.08, `⏎⏎` 0.07, `I` 0.05, `E` 0.03, `Y` 0.03
- **rule 12**: wins 7.7%, mean firing 0.212
  - fires on contexts ending in `g`, `c`, `"`; strongest: `out·a·big·laug` · `·high·and·caug` · `hands·and·laug` · `He·almost·caug` · `layed·and·laug`
  - gain where it wins: +3.19 ± 1.36; THEN raise `h` 0.25, `'` 0.21, `g` 0.06, `o` 0.05, `·` 0.05, `C` 0.03; when the gain is negative, raise `"` 0.22, `F` 0.14, `T` 0.10, `d` 0.07, `f` 0.06, `m` 0.06
- **rule 28**: wins 6.6%, mean firing 0.269
  - fires on contexts ending in `v`, `j`, `'`; strongest: `ball,·and·a·ye` · `ll·something·v` · `·was·getting·v` · `·was·feeling·v` · `se·pizza·and·j`
  - gain where it wins: +1.27 ± 1.95; THEN raise `"` 0.65, `·` 0.18, `t` 0.13, `d` 0.02, `a` 0.01, `�` 0.01; when the gain is negative, raise `H` 0.12, `Y` 0.10, `V` 0.05, `_` 0.04, ``` 0.04, `{` 0.04

## Layer 0 sequence mixer, head 0: 16 rules (crispness 0.22; uniform = 0.06)

- **rule 4**: wins 16.6% of positions, mean membership 0.123
  - stores contexts ending in: `a` (90), `v` (10)
  - strongest: `·her·toy·box·a` · `hook·the·box·a` · `ened·the·box·a` · `k·in·the·box·a` · `very·wealthy·a`
- **rule 15**: wins 15.7% of positions, mean membership 0.099
  - stores contexts ending in: `·` (59), `o` (41)
  - strongest: `·told·him,·"Bo` · `·Max·said,·"Bo` · `·and·said,·"Wo` · `·and·said,·"Wo` · `amed·Bobby.·Bo`
- **rule 8**: wins 14.7% of positions, mean membership 0.087
  - stores contexts ending in: `s` (84), `a` (12), `d` (4)
  - strongest: `He·pushed·hims` · `opcorn·by·hims` · `ould·feel·hims` · `·proud·of·hims` · `·proud·of·hims`
- **rule 14**: wins 13.0% of positions, mean membership 0.106
  - stores contexts ending in: `o` (53), `·` (45), `i` (2)
  - strongest: `·told·him,·"Bo` · `of·yummy·bambo` · `·Max·said,·"Bo` · `d·to·eat·bambo` · `le·playing,·Bo`
- **rule 12**: wins 10.0% of positions, mean membership 0.117
  - stores contexts ending in: `g` (56), `O` (21), `'` (19)
  - strongest: `dependent·is·g` · `ng."↵Bob·was·g` · `·The·jug·was·g` · `e.·Rocky·was·g` · `hen,·Joy·was·g`
- **rule 2**: wins 8.3% of positions, mean membership 0.091
  - stores contexts ending in: `·` (91), `y` (9)
  - strongest: `n·a·branch.↵A·` · `the·button.↵A·` · `on·a·table.·A·` · `gly·it·was.↵A·` · `d·happened.·A·`

## Layer 0 sequence mixer, head 1: 16 rules (crispness 0.36; uniform = 0.06)

- **rule 10**: wins 65.8% of positions, mean membership 0.342
  - stores contexts ending in: `M` (52), `N` (23), `·` (14)
  - strongest: `ojo·asked.·↵"N` · `,·mix,·mix!"↵M` · `frustrated.⏎⏎↵M` · `d·listen.·↵·↵F` · `ey·saw·1,·2,·3`
- **rule 8**: wins 10.1% of positions, mean membership 0.290
  - stores contexts ending in: `L` (53), `q` (24), `c` (18)
  - strongest: `firemen·came·q` · `·can·both·be·q` · `.·We·will·be·q` · `aid,·"Hello,·q` · `ut·please·be·q`
- **rule 12**: wins 8.5% of positions, mean membership 0.038
  - stores contexts ending in: `k` (75), `p` (18), `b` (4)
  - strongest: `s·about·to·tak` · `ou·have·to·tak` · `as·okay·to·tak` · `ady!·Let's·tak` · `·wanted·to·tak`
- **rule 6**: wins 7.3% of positions, mean membership 0.089
  - stores contexts ending in: `t` (66), `e` (25), `w` (8)
  - strongest: `me·popcorn.↵At` · `an·and·ran.↵At` · `th·her·mom.·At` · `itchen·with·ot` · `hugged·each·ot`
- **rule 3**: wins 3.5% of positions, mean membership 0.053
  - stores contexts ending in: `c` (100)
  - strongest: `re·with·long·c` · `nd·something·c` · `ed·something·c` · `nd·something·c` · `·was·getting·c`
- **rule 14**: wins 3.1% of positions, mean membership 0.032
  - stores contexts ending in: `k` (82), `p` (12), `b` (6)
  - strongest: `amous.·He·took` · `·plan.·He·took` · `·saw·the·crumb` · `·off·the·crumb` · `l·she·was·soak`

## Layer 0 sequence mixer, head 2: 16 rules (crispness 0.33; uniform = 0.06)

- **rule 5**: wins 31.9% of positions, mean membership 0.232
  - stores contexts ending in: `t` (100)
  - strongest: `udied·the·pict` · `·It·had·a·pict` · `and·had·a·pict` · `y·to·make·mist` · `dirty·and·filt`
- **rule 2**: wins 21.5% of positions, mean membership 0.116
  - stores contexts ending in: `v` (100)
  - strongest: `·ran·away·v` · `day,·Max·was·v` · `e·lawyer·was·v` · `ach.·Tom·was·v` · `se!"·Tom·was·v`
- **rule 4**: wins 21.4% of positions, mean membership 0.155
  - stores contexts ending in: `L` (55), `a` (39), `↵` (4)
  - strongest: `ll.↵Jack:·���L` · `nk·you,·I·am·L` · `e,·"Hi,·I·am·L` · `L` · `down·and·saw·L`
- **rule 0**: wins 3.0% of positions, mean membership 0.041
  - stores contexts ending in: `,` (45), `o` (28), `I` (27)
  - strongest: `eryone·giggle,` · `ad·an·idea!·"I` · `to·her·mom,·"I` · `ax·replied,·"I` · `ot·replied,·"I`
- **rule 11**: wins 2.9% of positions, mean membership 0.041
  - stores contexts ending in: `o` (100)
  - strongest: `l·rose.·The·ro` · `iends?"·The·ro` · `cked·up·the·ro` · `smooth·rock·ro` · `ng·down·the·ro`
- **rule 1**: wins 2.9% of positions, mean membership 0.044
  - stores contexts ending in: `l` (68), `o` (32)
  - strongest: `h.·The·jug·fel` · `t·by.·Jojo·fel` · `became·a·livel` · `re·was·a·livel` · `·a·monkey·befo`

## Layer 0 sequence mixer, head 3: 16 rules (crispness 0.58; uniform = 0.06)

- **rule 9**: wins 57.0% of positions, mean membership 0.451
  - stores contexts ending in: `a` (100)
  - strongest: `e·was·a·firema` · `ima` · `day·on,·Max·ma` · `eaf·clover,·ma` · `"Jerry!·You·ma`
- **rule 12**: wins 11.2% of positions, mean membership 0.123
  - stores contexts ending in: `t` (100)
  - strongest: `put·it."·Amy·t` · `adio.·You're·t` · `rave·and·try·t` · `d·goodbye·t` · `said·goodbye·t`
- **rule 2**: wins 10.7% of positions, mean membership 0.144
  - stores contexts ending in: `·` (100)
  - strongest: `door·and·Jack·` · `ay·cake!·Jack·` · `ty·hats.·Jack·` · `·garden.·Jack·` · `and·then·pick·`
- **rule 11**: wins 9.6% of positions, mean membership 0.098
  - stores contexts ending in: `T` (30), `S` (30), `B` (23)
  - strongest: `iends·called·R` · `hanked·R` · `forever,·and·R` · `ce·dog·named·R` · `n·R`
- **rule 3**: wins 3.7% of positions, mean membership 0.026
  - stores contexts ending in: `s` (89), `i` (10), `m` (1)
  - strongest: `oven.·They·wai` · `waited·and·wai` · `hungry·and·wai` · `nds.·They·swam` · `m·and·Lily·wai`
- **rule 15**: wins 2.0% of positions, mean membership 0.037
  - stores contexts ending in: `y` (100)
  - strongest: `them·a·new·way` · `way` · `ange·their·way` · `airy·flew·away` · `·and·flew·away`

## Layer 0 sequence mixer, head 4: 16 rules (crispness 0.50; uniform = 0.06)

- **rule 14**: wins 45.7% of positions, mean membership 0.330
  - stores contexts ending in: `t` (100)
  - strongest: `ice·pop.·Max·t` · `ack·your·bag·t` · `re·done,·Max·t` · `hake·the·box·t` · `oor·and·Jack·t`
- **rule 10**: wins 19.2% of positions, mean membership 0.185
  - stores contexts ending in: `i` (100)
  - strongest: `frustrated.↵Mi` · `·had·made·a·bi` · `d·the·magic·bi` · `was·a·magic·bi` · `.·The·magic·bi`
- **rule 2**: wins 14.8% of positions, mean membership 0.165
  - stores contexts ending in: `Y` (49), `c` (42), `p` (7)
  - strongest: `���↵Jill:·���Y` · `ojo·asked.·↵"Y` · `en·and·Lily.·Y` · `en·and·Lily.·Y` · `f·you,·Lily.·Y`
- **rule 5**: wins 8.8% of positions, mean membership 0.066
  - stores contexts ending in: `o` (100)
  - strongest: `the·tie·back·o` · `ake·one·back·o` · `e·shell·back·o` · `nd·went·back·o` · `iked·to·rock·o`
- **rule 13**: wins 3.6% of positions, mean membership 0.079
  - stores contexts ending in: `a` (100)
  - strongest: `.·The·shelf·ha` · `·a·ball·and·ha` · `lt·cool·and·ha` · `ry·tall·and·ha` · `he·root·and·ha`
- **rule 7**: wins 3.5% of positions, mean membership 0.034
  - stores contexts ending in: `L` (58), `q` (34), `↵` (6)
  - strongest: `e·park.·The·sq` · `·mouth.·The·sq` · `kay.·A·kind·sq` · `ed·the·fake·sq` · `er·new·fake·sq`

## Layer 0 sequence mixer, head 5: 16 rules (crispness 0.45; uniform = 0.06)

- **rule 4**: wins 42.0% of positions, mean membership 0.323
  - stores contexts ending in: `o` (96), `i` (4)
  - strongest: `re·for·you.·Wo` · `and·Bo` · `eir·mom·and·Bo` · `·named·Bob.·Bo` · `·Bob.·Bo`
- **rule 14**: wins 16.2% of positions, mean membership 0.133
  - stores contexts ending in: `�` (39), `·` (33), `⏎⏎` (23)
  - strongest: `s·celebrate!��` · `arm·me·up?�` · `e!��` · `arm·me·up?��` · `nd·the·door!��`
- **rule 9**: wins 12.7% of positions, mean membership 0.107
  - stores contexts ending in: `"` (83), `j` (8), `!` (3)
  - strongest: `its·name·was·"` · `·the·door·is·j` · `alled·it·his·"` · `·shiny·it·is!"` · `I·like·jokes!"`
- **rule 8**: wins 10.1% of positions, mean membership 0.040
  - stores contexts ending in: `·` (100)
  - strongest: `awyer's·face!·` · `anyone·there?·` · `n't·find·you,·` · `·pizza·place?·` · `proud·of·you,·`
- **rule 3**: wins 7.6% of positions, mean membership 0.084
  - stores contexts ending in: `f` (64), `!` (11), `,` (6)
  - strongest: `·to·grow·hair!` · `hind·the·door!` · `ny·face·on·it!` · `nt·to·hear·it?` · `everyone·was·d`
- **rule 10**: wins 6.0% of positions, mean membership 0.075
  - stores contexts ending in: `d` (100)
  - strongest: `s.·You·d` · `nd·Lily.·You·d` · `y·said,·"You·d` · `What·are·you·d` · `"Why·did·you·d`

## Layer 0 sequence mixer, head 6: 16 rules (crispness 0.55; uniform = 0.06)

- **rule 2**: wins 48.2% of positions, mean membership 0.333
  - stores contexts ending in: `·` (90), `d` (7), `L` (2)
  - strongest: `self·and·slid·` · `ve·her·eyes·-·` · `de.·They·slid·` · `.·But·he·slid·` · `.·Lee.·We·did·`
- **rule 11**: wins 35.8% of positions, mean membership 0.370
  - stores contexts ending in: `o` (95), `y` (3), `m` (2)
  - strongest: `t·smelly·anymo` · `·with·it·anymo` · `·so·lazy·anymo` · `ited.↵Tim's·mo` · `ot·upset·anymo`
- **rule 6**: wins 6.0% of positions, mean membership 0.047
  - stores contexts ending in: `m` (38), `i` (35), `z` (18)
  - strongest: `nt·cheese·pizz` · `·for·your·pizz` · `e·has·our·pizz` · `iver·your·pizz` · `·to·order·pizz`
- **rule 15**: wins 3.7% of positions, mean membership 0.044
  - stores contexts ending in: `m` (95), `?` (4), `i` (1)
  - strongest: `are·you·doing?` · `were·sleeping?` · `re·you·hiding?` · `·you·fighting?` · `named·Tim.·Tim`
- **rule 5**: wins 1.6% of positions, mean membership 0.023
  - stores contexts ending in: `s` (65), `l` (23), `e` (10)
  - strongest: `s!"·Mr.·Nice·l` · `,·colorful·ros` · `y·colorful·ros` · `y·said,·"You·s` · `u,·frog!·You·s`
- **rule 1**: wins 1.1% of positions, mean membership 0.019
  - stores contexts ending in: `f` (72), `e` (13), `k` (13)
  - strongest: `boxes·away.·Af` · `els·were·amaze` · `he·hot·day.·Af` · `er·with·it.·Af` · `smelly·pot.·Af`

## Layer 0 sequence mixer, head 7: 16 rules (crispness 0.40; uniform = 0.06)

- **rule 15**: wins 45.1% of positions, mean membership 0.236
  - stores contexts ending in: `i` (60), `!` (20), `↵` (10)
  - strongest: `led·it·his·"mi` · `ng·so·much·noi` · `·"Hello,·bird!` · `e·day.·The·noi` · `noise!·The·noi`
- **rule 1**: wins 17.0% of positions, mean membership 0.179
  - stores contexts ending in: `e` (71), `o` (29)
  - strongest: `e·at·night.·So` · `py·like·it.↵So` · `e·to·play!"·So` · `,"·he·said.↵So` · `m·the·tree.·So`
- **rule 8**: wins 14.7% of positions, mean membership 0.111
  - stores contexts ending in: `a` (97), `r` (2), `o` (1)
  - strongest: `g·room·with·ma` · `l·town·with·ma` · `·place·with·ma` · `fake·squash·ma` · `back!"·↵Rob·ra`
- **rule 5**: wins 7.0% of positions, mean membership 0.101
  - stores contexts ending in: `c` (55), `s` (24), `l` (13)
  - strongest: `,·I·can·go·bac` · `e·fake·one·bac` · `e·bird·get·bac` · `t·her·book·bac` · `l·and·went·bac`
- **rule 13**: wins 5.5% of positions, mean membership 0.082
  - stores contexts ending in: `t` (65), `h` (35)
  - strongest: `·soaked·from·h` · `ed·away·from·h` · `ll·away·from·h` · `"I·am·Apple,·t` · `ter·a·while,·t`
- **rule 3**: wins 4.4% of positions, mean membership 0.080
  - stores contexts ending in: `n` (89), `f` (11)
  - strongest: `boxes·away.·Af` · `he·hot·day.·Af` · `cream·away.↵Af` · `very·happy.↵Af` · `s·so·yummy!·Af`

## Layer 1 TSK feed-forward (first-order, raw firing): 36 rules, 28 win > 0.1% of positions

- **rule 10**: wins 20.7%, mean firing 0.188
  - fires on contexts ending in `·`; strongest: `,·"Mum,·can·I·` · `"Mommy,·can·I·` · `d·full.·Can·I·` · `·purrs·when·I·` · `e!���↵The·two·`
  - gain where it wins: -2.46 ± 1.16; THEN raise `x` 0.48, `'` 0.06, `⏎⏎` 0.05, `=` 0.03, `[` 0.02, `8` 0.02; when the gain is negative, raise `l` 0.25, `i` 0.19, `w` 0.14, `d` 0.11, `h` 0.07, `s` 0.07
- **rule 32**: wins 14.4%, mean firing 0.125
  - fires on contexts ending in `y`, `e`, `d`; strongest: `·forgot·my·toy` · `"I·lost·my·toy` · `f·had·many·toy` · `·with·many·toy` · `m·took·his·toy`
  - gain where it wins: -2.44 ± 1.41; THEN raise `F` 0.41, `A` 0.08, `8` 0.03, `=` 0.03, `$` 0.02, `%` 0.02; when the gain is negative, raise `s` 0.38, `y` 0.35, `t` 0.21, `n` 0.01, `m` 0.01, `o` 0.01
- **rule 12**: wins 12.8%, mean firing 0.144
  - fires on contexts ending in `s`, `k`, `d`; strongest: `·and·said,·"Hi` · `·Sue·said,·"Hi` · `ting·on·a·lily` · `all·people·mus` · `·the·rose,·"Hi`
  - gain where it wins: +0.79 ± 1.38; THEN raise `k` 0.28, `a` 0.27, `i` 0.15, `h` 0.10, `t` 0.08, `·` 0.03; when the gain is negative, raise `G` 0.10, `[` 0.06, `#` 0.05, `9` 0.04, `F` 0.04, `Y` 0.04
- **rule 4**: wins 7.0%, mean firing 0.157
  - fires on contexts ending in `u`, `o`, `x`; strongest: `.·He·was·so·ex` · `andy·was·so·ex` · `rfly·was·so·ex` · `Jojo·was·so·ex` · `·Tom·was·so·ex`
  - gain where it wins: +0.34 ± 1.54; THEN raise `u` 0.51, `y` 0.32, `i` 0.03, `I` 0.03, `o` 0.02, `r` 0.01; when the gain is negative, raise `c` 0.13, `?` 0.08, `q` 0.05, `p` 0.05, `Z` 0.04, `5` 0.04
- **rule 33**: wins 6.6%, mean firing 0.145
  - fires on contexts ending in `a`, `o`; strongest: `d·and·went·awa` · `One·da` · `e·very·dangero` · `ou·alone,"·Mia` · `·a·snack."·Mia`
  - gain where it wins: +1.75 ± 0.98; THEN raise `z` 0.26, `'` 0.20, `y` 0.17, `x` 0.09, `c` 0.03, `g` 0.03; when the gain is negative, raise `B` 0.35, `h` 0.23, `L` 0.14, `M` 0.06, `I` 0.05, `A` 0.03
- **rule 7**: wins 4.8%, mean firing 0.131
  - fires on contexts ending in `n`, `i`; strongest: `"No,·it·is·min` · `le·puppy!"·sai` · `dle·in·his·min` · `ig·rolling·pin` · `hey·went·up·an`
  - gain where it wins: +2.89 ± 0.99; THEN raise `d` 0.77, `a` 0.09, `y` 0.04, `·` 0.03, `i` 0.01, `s` 0.01; when the gain is negative, raise `B` 0.16, `7` 0.05, `=` 0.05, `/` 0.04, `H` 0.03, `[` 0.03

## Layer 1 sequence mixer, head 0: 16 rules (crispness 0.49; uniform = 0.06)

- **rule 11**: wins 59.4% of positions, mean membership 0.325
  - stores contexts ending in: `f` (99), `n` (1)
  - strongest: `·said,·"This·f` · `·Amy·thought·f` · `Lily·thought·f` · `Lily·thought·f` · `'t·find·you,·f`
- **rule 5**: wins 23.0% of positions, mean membership 0.183
  - stores contexts ending in: `.` (50), `e` (35), `t` (12)
  - strongest: `nted·to·go·up.` · `o·wake·Tom·up.` · `ver·giving·up.` · `en·he·woke·up.` · `it·up.`
- **rule 9**: wins 10.5% of positions, mean membership 0.236
  - stores contexts ending in: `l` (96), `y` (3), `G` (1)
  - strongest: `·it·rained·a·l` · `und·became·a·l` · `w.·He·made·a·l` · `was·a·l` · `x,·it·made·a·l`
- **rule 8**: wins 2.1% of positions, mean membership 0.036
  - stores contexts ending in: `d` (68), `y` (17), `"` (6)
  - strongest: `m.↵One·sunny·d` · `t.↵One·sunny·d` · `d.↵One·sunny·d` · `One·sunny·d` · `One·sunny·d`
- **rule 1**: wins 2.0% of positions, mean membership 0.058
  - stores contexts ending in: `h` (66), `u` (16), `i` (12)
  - strongest: `g·trees.·And·E` · `·read·about·fu` · `orn·was·not·fu` · `Lily·says.↵"Bu` · `The·jar·was·fu`
- **rule 3**: wins 1.2% of positions, mean membership 0.009
  - stores contexts ending in: `s` (25), `a` (23), `o` (13)
  - strongest: `ot·tried·to·us` · `ed·to·us` · `nd·followed·as` · `was·bi` · `·the·object·as`

## Layer 1 sequence mixer, head 1: 16 rules (crispness 0.41; uniform = 0.06)

- **rule 5**: wins 34.8% of positions, mean membership 0.210
  - stores contexts ending in: `t` (44), `c` (18), `y` (18)
  - strongest: `ent·to·Mr.·Nic` · `t·is!"·Mr.·Nic` · `ch·let·Mr.·Nic` · `t·when·Mr.·Nic` · `riend,·Mr.·Nic`
- **rule 4**: wins 20.0% of positions, mean membership 0.143
  - stores contexts ending in: `o` (41), `r` (31), `n` (20)
  - strongest: `saw·Tom·had·kn` · `is·mom,·not·kn` · `g·rock!↵Tom·fe` · `saw·her·and·kn` · `Tom·did·not·kn`
- **rule 11**: wins 17.9% of positions, mean membership 0.155
  - stores contexts ending in: `u` (31), `l` (15), `w` (14)
  - strongest: `play."↵Mom·joi` · `,·and·Dave·joi` · `·park·also·joi` · `.·He·did·not·k` · `.·He·did·not·k`
- **rule 6**: wins 9.2% of positions, mean membership 0.071
  - stores contexts ending in: `i` (39), `o` (15), `h` (15)
  - strongest: `it·visited.·↵E` · `·mom·said,·"Li` · `pset,·Elly?"·E` · `·had·three·chi` · `·house·and·shu`
- **rule 2**: wins 6.3% of positions, mean membership 0.060
  - stores contexts ending in: `e` (94), `h` (5), `o` (1)
  - strongest: `side.·When·she` · `.·But·when·she` · `riendly!",·she` · `thing·that·she` · `.↵But·then·she`
- **rule 13**: wins 2.9% of positions, mean membership 0.037
  - stores contexts ending in: `·` (42), `.` (19), `y` (14)
  - strongest: `s,·I'm·cooking` · `nize·the·face.` · `he·tree·to·res` · `nd·said,·"You'` · `ey·said,·"You'`

## Layer 1 sequence mixer, head 2: 16 rules (crispness 0.49; uniform = 0.06)

- **rule 7**: wins 58.6% of positions, mean membership 0.430
  - stores contexts ending in: `m` (42), `,` (39), `u` (7)
  - strongest: `ow·to·make·yum` · `That·looks·yum` · `·Sue·got·a·yum` · `·tree·with·yum` · `·It·was·so·yum`
- **rule 10**: wins 9.7% of positions, mean membership 0.064
  - stores contexts ending in: `s` (96), `o` (4)
  - strongest: `of·it.·Tim·was` · `on·it.·Tim·was` · `again.·Tim·was` · `d·Tim.·Tim·was` · `o·run.·Tom·was`
- **rule 12**: wins 9.7% of positions, mean membership 0.055
  - stores contexts ending in: `·` (48), `n` (43), `h` (6)
  - strongest: `·Tom·jumped·on` · `frog·jumped·on` · `o·he·jumped·on` · `o·he·jumped·on` · `·is·a·brillian`
- **rule 5**: wins 5.3% of positions, mean membership 0.055
  - stores contexts ending in: `l` (65), `k` (27), `t` (5)
  - strongest: `ig,·soft·blank` · `d·a·little·inf` · `ve·him·a·blank` · `t·there·is·onl` · `ired·and·crank`
- **rule 0**: wins 3.6% of positions, mean membership 0.044
  - stores contexts ending in: `a` (83), `n` (14), `u` (3)
  - strongest: `snacks·and·ran` · `It·ran·and·ran` · `er·mom·and·ran` · `s·bags·and·ran` · `·coats·and·ran`
- **rule 3**: wins 3.2% of positions, mean membership 0.059
  - stores contexts ending in: `n` (77), `c` (7), `y` (6)
  - strongest: `ed·to·rescue·L` · `nd·it·became·L` · `.·They·all·enj` · `un·books·for·L` · `ve·and·patienc`

## Layer 1 sequence mixer, head 3: 16 rules (crispness 0.62; uniform = 0.06)

- **rule 2**: wins 33.6% of positions, mean membership 0.297
  - stores contexts ending in: `·` (45), `w` (10), `o` (9)
  - strongest: `nd·it·became·L` · `ed·to·rescue·L` · `u."·She·gave·T` · `firemen·gave·T` · `·spirit·gave·T`
- **rule 15**: wins 29.9% of positions, mean membership 0.266
  - stores contexts ending in: `e` (100)
  - strongest: `be·a·good·frie` · `came·good·frie` · `came·good·frie` · `came·good·frie` · `came·good·frie`
- **rule 8**: wins 17.4% of positions, mean membership 0.169
  - stores contexts ending in: `a` (98), `O` (1), `k` (1)
  - strongest: `e·face,·"Who·a` · `ed·him,·"Who·a` · `p,·we·can·do·a` · `·When·we·try·a` · `ank·you,·you·a`
- **rule 12**: wins 10.9% of positions, mean membership 0.068
  - stores contexts ending in: `n` (97), `s` (2), `l` (1)
  - strongest: `Jessica·couldn` · `rd·that·couldn` · `·but·he·couldn` · `ted·and·couldn` · `dge·and·couldn`
- **rule 9**: wins 5.3% of positions, mean membership 0.110
  - stores contexts ending in: `o` (39), `a` (27), `y` (24)
  - strongest: `nd·his·mom·any` · `eety` · `·you·a·favor?"` · `there?·Can·any` · `·take."·Tweety`
- **rule 5**: wins 0.9% of positions, mean membership 0.013
  - stores contexts ending in: `t` (43), `y` (26), `h` (20)
  - strongest: `'m·on·Maple·St` · `on·the·queen's` · `ld·the·queen's` · `as·not·like·ot` · `was·a·big·butt`

## Layer 1 sequence mixer, head 4: 16 rules (crispness 0.56; uniform = 0.06)

- **rule 8**: wins 80.6% of positions, mean membership 0.537
  - stores contexts ending in: `p` (100)
  - strongest: `·went·to·the·p` · `n·the·smelly·p` · `ith·a·pretty·p` · `·sad·for·the·p` · `·He·was·very·p`
- **rule 13**: wins 5.7% of positions, mean membership 0.060
  - stores contexts ending in: `n` (66), `u` (25), `o` (9)
  - strongest: `ith·lots·of·yu` · `was·full·of·yu` · `was·full·of·yu` · `had·never·seen` · `hen·smelled·yu`
- **rule 0**: wins 3.7% of positions, mean membership 0.072
  - stores contexts ending in: `a` (38), `e` (27), `o` (13)
  - strongest: `as·a·big·yello` · `uld·make·diffe` · `kitchen·smelle` · `down·and·lande` · `tree·and·lande`
- **rule 15**: wins 2.8% of positions, mean membership 0.023
  - stores contexts ending in: `n` (75), `i` (25)
  - strongest: `ran·outside·in` · `nd·a·little·in` · `d·the·radio·in` · `ther·figure·in` · `nd.·He·fell·in`
- **rule 5**: wins 2.3% of positions, mean membership 0.103
  - stores contexts ending in: `a` (100)
  - strongest: `·Bob.·They·sha` · `�↵Jill:·���Tha` · `ore·fun·to·sha` · `·happy·and·sha` · `smiled·and·sha`
- **rule 9**: wins 1.7% of positions, mean membership 0.035
  - stores contexts ending in: `·` (93), `B` (5), `z` (1)
  - strongest: `·were·waiting·` · `t·his·journey·` · `prise·waiting·` · `wanted·to·buy·` · `wanted·to·buy·`

## Layer 1 sequence mixer, head 5: 16 rules (crispness 0.40; uniform = 0.06)

- **rule 4**: wins 51.8% of positions, mean membership 0.348
  - stores contexts ending in: `t` (73), `m` (9), `p` (7)
  - strongest: `ound·a·long·st` · `The·balloon·st` · `d·it·up·and·st` · `not·to·kick·st` · `d·water·and·st`
- **rule 5**: wins 24.7% of positions, mean membership 0.111
  - stores contexts ending in: `n` (64), `I` (28), `A` (4)
  - strongest: `hout·them.·Can` · `one·there?·Can` · `nored·you.·Can` · `.↵"We·both·wan` · `.·The·bug·wan`
- **rule 6**: wins 4.1% of positions, mean membership 0.063
  - stores contexts ending in: `o` (30), `u` (23), `m` (23)
  - strongest: `igure·too?"↵Mo` · `aid·to·Ben,·"O` · `s·mom·said,·"O` · `"↵Mom·said,·"O` · `d·and·said,·"O`
- **rule 11**: wins 3.6% of positions, mean membership 0.088
  - stores contexts ending in: `i` (45), `O` (36), `a` (11)
  - strongest: `own·puppy.·↵"O` · `edle·and·put·i` · `·Ben·and·put·i` · `,·"Let's·put·i` · `h·the·radio.↵O`
- **rule 0**: wins 3.5% of positions, mean membership 0.044
  - stores contexts ending in: `h` (94), `·` (4), `v` (1)
  - strongest: `the·jug·like·h` · `the·jug·like·h` · `h·each·piece·h` · `med·Mia.·Mia·h` · `·a·very·nice·h`
- **rule 14**: wins 3.2% of positions, mean membership 0.042
  - stores contexts ending in: `n` (61), `·` (25), `x` (13)
  - strongest: `loud·noise,·an` · `o·the·rose,·an` · `u·can·take,·an` · `pretty·one,·an` · `es,·a·nose,·an`

## Layer 1 sequence mixer, head 6: 16 rules (crispness 0.51; uniform = 0.06)

- **rule 9**: wins 44.3% of positions, mean membership 0.265
  - stores contexts ending in: `d` (56), `e` (37), `w` (7)
  - strongest: `l·and·relieved` · `e.·One·day,·Tw` · `t·you·take,·Tw` · `·not·take."·Tw` · `aid,·"You·save`
- **rule 4**: wins 34.7% of positions, mean membership 0.273
  - stores contexts ending in: `s` (67), `i` (15), `y` (14)
  - strongest: `her·how·ugly·i` · `e·a·joke.·Do·y` · `·woke·up·and·s` · `arned·a·very·i` · `ed·the·radio·i`
- **rule 11**: wins 11.2% of positions, mean membership 0.138
  - stores contexts ending in: `i` (52), `s` (14), `k` (12)
  - strongest: `·quiet·and·res` · `ly·for·the·res` · `er·for·the·res` · `get·scared.·Ti` · `d·they·did.·Ti`
- **rule 2**: wins 4.1% of positions, mean membership 0.045
  - stores contexts ending in: `m` (47), `y` (23), `f` (8)
  - strongest: `rywhere.↵Fluff` · `he·banana,·Tom` · `·at·the·bottom` · `·at·the·bottom` · `Sue·said,·"Tom`
- **rule 6**: wins 2.4% of positions, mean membership 0.087
  - stores contexts ending in: `g` (64), `L` (18), `q` (17)
  - strongest: `nd·it·became·L` · `·can·both·be·q` · `down·and·saw·L` · `·there·was·a·q` · `·there·was·a·q`
- **rule 3**: wins 1.3% of positions, mean membership 0.039
  - stores contexts ending in: `e` (38), `h` (15), `i` (12)
  - strongest: `,·"Can·I·touch` · `n.·Can·I·touch` · `ised·and·touch` · `Mr.·Nice·touch` · `ng.·Lily·touch`

## Layer 1 sequence mixer, head 7: 16 rules (crispness 0.76; uniform = 0.06)

- **rule 8**: wins 37.8% of positions, mean membership 0.363
  - stores contexts ending in: `h` (30), `d` (26), `o` (25)
  - strongest: `others·than·to` · `pon·a·time,·th` · `dropped·the·to` · `·couldn't·find` · `nd·get·your·to`
- **rule 3**: wins 33.2% of positions, mean membership 0.278
  - stores contexts ending in: `↵` (52), `·` (48)
  - strongest: `ee·and·slept.↵` · `eep·it·clean.↵` · `king·so·long.↵` · `was·the·best.↵` · `·green·field.↵`
- **rule 4**: wins 9.9% of positions, mean membership 0.090
  - stores contexts ending in: `.` (60), `t` (40)
  - strongest: `ud·of·himself.` · `ud·of·himself.` · `rn·by·himself.` · `because·I·am·t` · `carry·Max·up·t`
- **rule 13**: wins 9.5% of positions, mean membership 0.076
  - stores contexts ending in: `n` (90), `a` (10)
  - strongest: `there·was·a·ta` · `ngs·you·can·ta` · `pple!·I·can·ta` · `s.·"You·can·ta` · `d:·"You·can·ta`
- **rule 11**: wins 5.6% of positions, mean membership 0.054
  - stores contexts ending in: `d` (98), `e` (2)
  - strongest: `d·books.·One·d` · `·rest·of·the·d` · `·rest·of·the·d` · `ny·toys.·One·d` · `th·toys.·One·d`
- **rule 7**: wins 1.4% of positions, mean membership 0.049
  - stores contexts ending in: `h` (32), `w` (20), `y` (16)
  - strongest: `eelings,·too.⏎⏎` · `·the·firemen·g` · `e·could·feel·h` · `ir!·The·hair·g` · `uld·play·fun·g`

