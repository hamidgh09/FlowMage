#!/usr/bin/gnuplot

### Input file
baseline_file = "csvs/Vanilla-FastclickOFRXPPS.csv"
ref_file = "csvs/FAJITAOFRXPPS.csv"
y_scale=1

### Margins
bm_bottom = 0.125
tm_bottom = 0.98
lm = 0.14
rm = 0.99

font_type = "Helvetica,28"
font_type_bold = "Helvetica-Bold,28"

set terminal pdf color enhanced font "Helvetica,38" size 10cm,9cm;
set style fill solid 1 border -1 
#set style boxplot nooutliers
#set style boxplot fraction 1.00
#set style data boxplot
#set boxwidth 1
# set size square

# Margins
set bmargin at screen bm_bottom
set tmargin at screen tm_bottom
set lmargin at screen lm
set rmargin at screen rm

# Legend
set key outside opaque bottom Right title
#set key top right opaque
set border back
set key box linestyle 1 lt rgb("#000000")
set key vertical maxrows 2
set key width -2.2
set key height 0.8
set key samplen 3.0
set key at 2.55, 86.95
set key font "Helvetica, 18"

#set key bottom Left left reverse box width 2
set xtics font "Helvetica, 18" 
set ytics font "Helvetica, 18"
# X-axis
set xlabel "Available CPU cores" font "Helvetica,19"
set xlabel offset 0,1.6
set xtics offset 0,0.7 nomirror
set xtics border in scale 1,0.5 norotate autojustify mirror
set xrange [-0.5:4.5]

# Y-axis
set ylabel "Throughput (Mpps)" font "Helvetica,19"
set ylabel offset 4.2,0
set yrange [0:108]
set ytic 10
set ytics offset 0.7,0 nomirror
set tic scale 0.2

set grid

set style data histogram
set style histogram cluster gap 1 
set xtics format ""
# set terminal pdf size 10cm,10cm
set output 'eval-cores-var-throughput.pdf'

plot baseline_file using 5:xtic(1) lc rgb '#2c7bb6' title "Vanilla-FastClick", \
ref_file using 5 with histogram lc rgb '#abd9e9' title "FAJITA"
