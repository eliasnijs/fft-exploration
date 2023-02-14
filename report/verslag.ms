.nr figstep 1 1
.TL
Practicum Fourier
.AU
Elias Nijs
.AI
Inleiding tot de telecommunicatie
Universiteit Gent
.DA
.AM
.2C
.R1
accumulate
database bib.ref
.R2
.EQ
delim $$
.EN
.nr figstep 1 1
.NH 1
Vraag 1 - Een signaal maken, en samplen
.LP
(enkel code)
.NH 1
Vraag 2 - Implementeren van de Fast-Fourier Transformatie
.LP
(enkel code)
.NH 1
Vraag 3 - Spelen met de frequenties
.LP
We beantwoorden enkele vragen met behulp van de vorige twee funties. We visualiseren elke vraag ook
met twee figuren. De eerste met het signaal in het tijdsdomein en de andere in het frequentiedomein.
vermits de FFT een complex getal is, zullen we de norm plotten.
.NH 2
.LP
.I
We geven een reeel signaal als input. Wat betekent dit in verband met de
frequenties die we van de FFT bekomen?
.LP
Het resultaat van de fft zal gespiegeld zijn rond de y as. Bovendien zullen we enkel waarden terugkrijgen
die tussen de negatieve Nyquist-frequentie en de Nyquist-frequentie liggen. De Nyquist frequentie is
gelijk aan de helft van de samplings frequentie.
.NH 2
.LP
.I
Maak een signaal van 2 Hz en plot het signaal zowel in het tijd- en frequentiedomein (dit mogen twee aparte figuren zijn). Wat zie je? Gebruik
hiervoor een sample-rate van 32 Hz en sample punten over 8 tijdseenheden
in het tijdsdomein.
.PSPIC -C "images/3b_timedomain.eps"
.PSPIC -C "images/3b_freqdomain.eps"
.LP
In het tijdsdomein zien we een sinus golf met een frequentie van 2 Hz en een amplitude van 1.
Dit betekent dat het signaal 2 keer per seconde oscilleert met een maximale waarde van 1.

In het frequentiedomein zien we een piek bij 2 Hz, wat correspondeert met de frequentie van het
tijdsdomein-signaal. De amplitude van de piek geeft aan hoe sterk de harmonische component met
frequentie 2 Hz is in het tijdsdomein-signaal.

In het frequentiedomein zien we ook een piek op -2 Hz. Dit komt omdat de FFT van een reeel
signaal symmetrisch is in het frequentiedomein, zoals uitgelegd bij de vorig vraag.
.NH 2
.LP
.I
Voeg nu (bij het vorige punt) twee signalen toe: een signaal van 3 Hz met
een fase van $ pi /2 $ en een signaal van 7 Hz. Wat zie je nu?
.PSPIC -C "images/3c_timedomain.eps"
.PSPIC -C "images/3c_freqdomain.eps"
.LP
Nu zien we in het tijdsdomein de som van de drie sinusfuncties.

In het frequentie domein zien we de frequentie van deze drie sinusfuncties terugkomen, we zien
op deze frequenties telkens een piek. Hoewel de amplitude van al deze sinus functies dezelfde is,
zien we toch een verschil in amplitude in het frequentie domein.
.NH 2
.LP
.I
Verlaag de sampling rate naar 16 Hz. Wat merk je nu? Verklaar.
.PSPIC -C "images/3d_timedomain.eps"
.PSPIC -C "images/3d_freqdomain.eps"
.LP
We merken dat het patroon in het tijdsdomein veranderd is. Het is veel hoekiger.

Op het frequentie domein kunnen we, hoewel deze breder geworden zijn, de pieken nog steeds
duidelijk terugzien en dus onze compositie terugvinden.

Dit komt doordat de sampling verlaagt is.
.NH 2
.LP
.I
Verlaag de sampling rate naar 8 Hz. Wat merk je nu? Verklaar.
.PSPIC -C "images/3e_timedomain.eps"
.PSPIC -C "images/3e_freqdomain.eps"
.LP
We zien complete vervorming van het signaal in het tijdsdomein.

In het frequentie domein zien we ook dat de pieken niet meer op de juiste plaatsen staan. We
kunnen dus de originele compositie niet terugvinden.

We leiden af dat de sampling rate op dit moment te laag is.

We verwijzen naar de Nyquist-frequentie, die gelijk is aan de helft van
de samplingsfrequentie.  Reele signalen bestaan alleen uit frequenties in het bereik
tot de Nyquist-frequntie terwijl complexe signalen frequenties kunnen bevatten in het hele
frequentiebereik van nul tot oneindig.
.NH 1
Vraag 4 - De FFT in de praktijk
.LP
We implementeren een equalizer. Hiermee bekijken we de volgende bestanden en bepalen hun
frequentiedomein:
.IP \[bu] 2
.BI "toonA.wav",
.PSPIC -C "images/toonA.eps"
.IP
We zien een enkele toon rond de $450$ Hz.
.IP \[bu]
.BI "toonB.wav",
.PSPIC -C "images/toonB.eps"
.IP
We zien een enkele toon op $1$ kHz.
.IP \[bu]
.BI "toonC.wav",
.PSPIC -C "images/toonC.eps"
.IP
We zien 2 tonen. De toon uit
.I toonB.wav
plus een lagere toon die zich op $100$ Hz bevindt.
.NH 2
Vraag 5 - DJ spelen
.LP
.I
Probeer daarna om een low-pass filter te gebruiken in je lievelingsmuziek om
instrumenten zoals de basgitaar en de drums weg te werken.
Welke frequenties heb je hiervoor weggewerkt?
.LP
Hiervoor werdern all frequenties lager dan 500Hz weggewerkt met de filter.
