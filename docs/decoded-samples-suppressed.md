# Decoded samples with adjacent-token suppression

Verbatim output of `runs/bench-adj.log` (2026-11-15, CPU, 10 threads),
i.e. the shipped default sampler: fixed mutable-slot quota, `remask_ratio=0.25`,
adjacent-collision suppression on.

```
python benchmark.py --checkpoint runs/bfg-lora/last.pt --num_prompts 12 \
  --prefix_len 128 --canvas_len 32 --batch_size 2 --steps 2 4 8 \
  --temps 0.0 0.3 0.5 --top_k 50 --remask_ratio 0.25 --show 12 --threads 10
```

Compare with `docs/decoded-samples.md`, which is the same checkpoint before
suppression and the quota fix.

## [0] Once upon a time, in a small coastal village, the fishermen noticed that

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
the fish the fish of the the the the the the the the the the fish of the the the the fish the the the the the the the the the the
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
the fish of the fish were not the the the fish of the fish fish the the the the the fish fish of the sea sea fish of the the the the
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
the were able able to to fish the fish of the the the the same of the fish were fish fish were not the the fish were not same the fish were
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
the fish of had the, a the to the and the, and and the that. the the. to a,,. the of. the and,
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
theerman had a problem. of catching and. The fish fish the the fish, a, a fish fish the. a day the fish, the the fish
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
the they were having a with fish caught in the and and. of the the were able they the the the fish were fish and the fish were the were of
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
the fish fish of were a a had were of the of of the fish. on the the fish the and the water a of a the in of of fish
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
the fish of a certain certain of the a, the fish the of the the fish are in the to the beach,, fish that are swimming in the current
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
the sea full calm the and sun average temperature of of 1 When And So they caught them fish in in the sea and they a great number of were caught
```

**causal generate**

```text
the sea was getting more and more salty. The sea was becoming more and more salty, and the fish were dying. The fishermen decided to do something about it

```

## [1] To configure the proxy, edit the settings file and set

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
the proxy proxy to proxy proxy











#













```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
the proxy to to proxy proxy



``#
#








#
#



#

```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
following proxy to desired proxy proxy. to to the the desired proxy. proxy to the proxy to proxy to the the to proxy proxy proxy the proxy to the proxy
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
the proxy the
 proxy proxy to the proxy proxy

` proxy ` to` the to the proxy the
 proxy
 configuration. to
 to
`
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
the proxy to proxy proxy
 proxy to proxy
,
 and proxy proxy the the proxy proxy proxy to, set proxy to the, you can to the proxy
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
the server to the proxy address port the port of the proxy.

 For example,,, set to server proxy of proxy address to port server which the proxy address
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
the "proxy proxy proxy proxy" "proxy"  proxy" proxy"proxy " proxy proxy if proxy  is/ is proxy (.8 - of.
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
the proxy command in to proxy -1 proxy proxy

If you to configure the, proxy the
 to proxy command, set the the proxycommand is is
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
following settings to desired proxy to be to``server  proxy proxy to ``

 if you to to be then the configuration the proxy is allowed to be used

```

**causal generate**

```text
the following parameters:

```
# Proxy settings
# Proxy server
# Proxy port
# Proxy protocol
# Proxy protocol
# Proxy protocol
# Proxy

```

## [2] ## Quick start

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
the



`











or





##

 











 







##
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
the



```
`



`

or



``##

Installation````

``





`
``





```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
following

````npmnpm`

````npm

`

```

 run the









```

`

``npm``



``




```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
the command.

. to

 the and to


 the the

 and,

 the,
 the,















```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
the script

``


 Install the

```



``.

 -

##````



``````

``








``
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
the script

``npm
 be install run the script script.

 the



``



  1
2  
 
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
the following

 to and for the


.
. install

 the to the



.

 and
 and the




.

 the

```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
the example using

``
 a single line, or and or

 run it`` the command or the

 a command, the

`` line
, then
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
the following```bash


`

```

##```
```

``.```



````

````````````.``


```

**causal generate**

```text
the following commands:

```bash
# Install the package
sudo apt-get install libglib2.0-dev

# Install the dependencies
sudo apt-get

```

## [3] {

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
by % 01",
",
",
",
 ",
",
",
",
",
",
",
",
",
",
",
",
",
",
",
",
",
",
",
",
",

```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
by 010%",
",
",
",
  "output": "",
",
",
",
 "",
",
 ",
 "",
",
",
 ",
",
",

```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
by %%%",
  "",
  "":output":QuarterQuarter",
 output":",
",
",
  "output":":Quarter",
",
",

```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
by  percent%5,0",
",
02",
  "",
000",
",
",
00",
  " "": ",
",
",

```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
by 10% (0%",
2",
  "output": "100000% 0",
  " 1 "output
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
$ ",
",
",
",
  "output",
  "output":",
   ":",
",
 ",
",
",
",
",
  " ",
 
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
by 6%% (15. ",
 0010 :00 =01 =:sum0:",
   
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
by 8.5%91004",
  "  " "506"
90",
 1",
  "",
  "
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
6123%",
",
",
  titleoutput":":QuarterRevenue",
",
",
",
  "outputoutput": "",
 " "outputoutput":
```

**causal generate**

```text
1.5% in the first quarter of 2019",
  "output": "Quarterly revenue rose 1.5% in the

```

## [4] - first, collect the measurements

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
calculate the mean


,,,,,,,,,,,,,,,,,,,,,,,,,,
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
calculate the outliers

- fourth the fourth the fourth,,, fifth, and,,,,,,, and and, the,,,,
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
calculate outliers


- fourth, calculate the outliers,,, fifth then, fourth, and,, and, and then and the fourth and,,
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
calculate the outliers and the the outliers,

,
,,
, and,, fourth,,,,,, the data
,
,
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
remove the outliers for the, and,,
- fourth,, the,,, the outliers, the, the,,,
, fifth,,
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
plot the outliers

- fourth plot the and finally save remove the clean data


- fifth,, and finally, remove clean, finally, and and
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
calculate the outliers and and and finally fourth,, plot,,,, the, the, and,- to and-
, the,,,,
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
calculate the outliers

- fourth, the outliers

-3rd outliers, remove them,
4, remove,,,, remove
,,
5
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
plot outliers
- fourth, remove plot the
- fourth plot plot the, plot,

###-
 fifth, plot
,
, the

```

**causal generate**

```text
calculate the mean and standard deviation
- fourth, plot the data
- fifth, plot the histogram
- sixth, plot the boxplot
- seventh,

```

## [5] The committee reviewed the proposal and concluded that

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
the proposal the the proposal of the the proposal of the the the the the the the the the the the the the the the the the the the the the the
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
the proposal the proposal is the the proposed proposed of the and the the the the the proposal the the the proposal the the of the the proposal the the the the
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
the proposal is not to the committee proposal. The proposal of proposal the proposal the the committee the committee of of of proposal of the committee. The and the the
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
the proposal the to the of. , the, the that to  the. The the of the. and and, to the  
. the
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
the proposal is appropriate,,..2, .......,2..0. 0.1.,..
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
proposed proposal was was the way of the and interest of the.. The proposal the proposal of of and. and.. The committee of the to that of
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
the proposed the that the and and to78 and are by,
... to of, of and the of
the to, the.

```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
it was the the that. The that the is the the of this proposal, that the that the the proposed was and to the that that the. The The
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
proposed proposal is a a good is a good and the a that it it. to.. The committee committee is the to the The and the is to
```

**causal generate**

```text
the proposed amendment to the Act would not be in the public interest. The committee also recommended that the proposed amendment be withdrawn.
The committee also recommended that the proposed

```

## [6] def merge_intervals(intervals):

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
   intervals = = intervals intervals intervals intervals intervals = intervals

 intervals intervals = intervals intervals intervals
"""


"""
"""
"""
def
"""
    intervals"""

```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
   intervals = = sorted intervals intervals = intervals intervals intervals = = intervals intervals = =
    intervals = intervals = sorted intervals intervals intervals intervals = intervals intervals intervals
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
   intervals = lambda sorted:
        sorted
 = sorted = sorted(lambda intervals

    sorted sorted(lambda


 = = = lambda



```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
   intervals = = sorted(int intervals intervals intervals
"""
"""
 intervals
"""
    intervals intervals intervals intervals intervals intervals intervals intervals"""
"""
 intervals
 a
 b"""

```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
  .sort = = sortedintervals,,
    intervals
    intervals = intervals intervals intervals intervals
       intervals intervals intervals intervals intervals intervals sorted intervals

```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
   return lambda lambda_intervals:
          
    intervals = sorted sorted intervals
   
    lambda intervals

        intervals = sorted

   

```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
   intervals = = intervalsint intervals sorted sorted by, intervals =,
    intervals sorted intervals intervals intervalsint
 intervalsint start new: = intervals
 intervals
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
   intervals = sorted intervals sorted, keysorted
 intervals.sort = intervalssorted, intervals


 intervals intervals = intervals

 intervals intervals = intervalssorted =
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
   = = sorted(int(intervals)
"""
    intervals   ()
 intervals intervals ="""
 =int = """
""" intervals intervals intervals)
    return intervals
"""
"""

```

**causal generate**

```text
   intervals.sort()
    merged = []
    for i in range(len(intervals)):
        if intervals[i][0] == intervals[i+1][0]:


```

## [7] In 1969, engineers at the laboratory succeeded in

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
producing a new of of of the the...     the first the the      1,     
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
producing the material that    1   The material was  was the the the first of these     00  1
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
producing a type of material that could of the type,, which the to be of of materials,,, and the,,,, and, 1
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
creating a new that to a to to the to, that, and so... the. This.. the of the and the, of ,
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
producing a the a new of   .0 The new was .. of,. of  of. 50,,  1
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
producing a material of the form  1   0.0 were able a was a way to  a a     The The was
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
producing the a  of  million atoms0 and 1 of8.. in that., and was,, of the, in were were, of
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
producing a the first new of 0. of 0.034 (  .  of0, 4.0 2 and the
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
producing a atom of a with the following  2. a) ( b) b the The b) b c) c c) b the b b
```

**causal generate**

```text
producing a new type of material that could be used to make a new type of computer chip. The new material was called silicon dioxide, and it was the first

```

## [8] Recipe: heat the pan, add the onions, and cook until

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
they is is...       1,,,,,, ,,,,,,,,,  
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
they is a...2 Add add the garlic onions, and, and add the,,, and , and, cook cook the garlic,.
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
they soft little golden brown Then Add add tomato sauce, and and and and cook it it and add the garlic until it is a. and add the tomato garlic
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
the is. the.. . the.
 , . ,. 
.  ,
      , 
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
they is,..  add the onions,., , ,,,,,,,, the onions are cook and, cook the,
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
they are,1. Add Add add chicken, bit,, and cook until 1 and  and  are needed is,, add 1.

2
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
they is to done done, add. when the onions is done you, cook the meat onion, add the  then add the to taste.
..,
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
they are is in.
, ,60, 1,2,,1,  1, , ,, 3 ,
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
they translucent (55).
Add the heat to and cook sugar pepper, and cook the mixture your your 1. add the salt and, and until the
```

**causal generate**

```text
the onions are soft and translucent.
2. Add the garlic and cook until fragrant.
3. Add the tomatoes and cook until the tomatoes are soft and the

```

## [9] The report argues that remote work changes urban economies because

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
it work work in and and,,,,,,,,, and, and,,,,,,,,,,,,,,
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
it makes workplaces more,,,,,,,,,, and,,,,,,,,,,, and,, and and
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
makes in lot of,,,, and,,,,,,, and,,, the, and,,,,,,, and,
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
it makes work is the in, people, and, the,,, and and,, and to the cities
,,.,,,,,
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
they
placesplaces the economy in in a,,, and,,,,, be,,, of,,,- and- the


```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
the need the for more for work, and the need of,,,, the, and,,, and the need and,, and and,,
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
, in the distance of work that,,, workers,, in to work,, people that
 the urban. and. ,, the people have
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text


of work the amount of the the numbers of and to,., of the the economy, the of., and, the in the that workers
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
remote work them people are become more competitive in world global world. This __________ The urban economy is is in changes in people the ability work and the need for
```

**causal generate**

```text
it creates new opportunities for people to work and to live in the same place. It also argues that remote work is a way to reduce the cost of commuting and

```

## [10] Once upon a time, in a small coastal village, the fishermen noticed that

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
the fish the fish of the the the the the the the the the the fish of the the the the fish the the the the the the the the the the
```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
the fish of the fish were not the the the fish of the fish fish the the the the the fish fish of the sea sea fish of the the the the
```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
the were able able to to fish the fish of the the the the same of the fish were fish fish were not the the fish were not same the fish were
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
the fish of a were a, the a to. the the, the fish of was.,.,. the the a to the a a was to
```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
the number of the were were very the the same the a fish fish. a and b a the the in the the sea. the fish of of the a
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
the fish they were much smaller in size the water than The They they caught fish fish and the fish 1 the the the the same all the the same size
```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
there were fish caught of of and fish the water and and the and a pair of, a of. that with a was the most a fish to in a
```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
the fish were eating the ocean and the the, the to to the the ocean the fish was not the the fish was the the a fish that the a fish
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
the sun was a little moon. They In it moon moon of moon, the the moon to the moon of, and the that moon water of it had an
```

**causal generate**

```text
the sea was getting more and more salty. The sea was becoming more and more salty, and the fish were dying. The fishermen decided to do something about it

```

## [11] To configure the proxy, edit the settings file and set

**bfg-diff k=2 T=0 d=0 f=0 r=0.25**

```text
the proxy proxy to proxy proxy











#













```

**bfg-diff k=4 T=0 d=0 f=0 r=0.25**

```text
the proxy to to proxy proxy



``#
#








#
#



#

```

**bfg-diff k=8 T=0 d=0 f=0 r=0.25**

```text
following proxy to desired proxy proxy. to to the the desired proxy. proxy to the proxy to proxy to the the to proxy proxy proxy the proxy to the proxy
```

**bfg-diff k=2 T=0.3 d=0 f=0 r=0.25**

```text
the proxy proxy to proxy
 the proxy to the the proxy proxy proxy
 to to to the proxy to proxy to to
  to/ to



```

**bfg-diff k=4 T=0.3 d=0 f=0 r=0.25**

```text
the    the proxy proxy proxy to the the proxy of to the proxy proxy proxy to the to the the the proxy to to the to the the proxy
 proxy
```

**bfg-diff k=8 T=0.3 d=0 f=0 r=0.25**

```text
following proxy to to 11 use the proxy to to proxy to to  to to the1 use
proxy to proxy to the to to to the

```

**bfg-diff k=2 T=0.5 d=0 f=0 r=0.25**

```text
the following the proxy proxy
 proxy
 proxy to,
,
 proxy the


) proxy
 
, of, or /proxy of proxy

```

**bfg-diff k=4 T=0.5 d=0 f=0 r=0.25**

```text
the port to listen

 address to listen the port in

    environment variable to

= =  for the  settings =
  proxy, port to listen
```

**bfg-diff k=8 T=0.5 d=0 f=0 r=0.25**

```text
the port in use the
proxy proxy server. The proxy uses listens on port of 7:80::7:8:8380
```

**causal generate**

```text
the following parameters:

```
# Proxy settings
# Proxy server
# Proxy port
# Proxy protocol
# Proxy protocol
# Proxy protocol
# Proxy
```
