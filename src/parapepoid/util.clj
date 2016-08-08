(ns parapepoid.util)

(defn select-values [m ks]
  (reduce #(conj %1 (%2 m)) [] ks))

(defn rand-range [min max]
  (+ min (rand (- max min))))

(defn rand-int-range [min max]
  (+ min (rand-int (- max min))))

(defn rand-magnitude [val percentage minimum maximum]
  (let [fraction (* val percentage)
        mini (Math/max (double minimum) (- val fraction))
        maxi (Math/min (double maximum) (+ val fraction))]
    (rand-range mini maxi)))

(defn rand-int-magnitude [val magnitude minimum maximum]
  (let [fraction (* (double val) magnitude)
        rounded-fraction (Math/round (Math/max fraction 1.0))
        mini (Math/max (long minimum) (- (long val) rounded-fraction))
        maxi (Math/min (long maximum) (+ (long val) rounded-fraction))]
    (rand-int-range mini maxi)))