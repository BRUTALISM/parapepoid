(ns parapepoid.approach.input-mappings
  (:require [parapepoid.util :as u]
            [thi.ng.color.core :as tc]))

; Various ways to map colors into NN inputs.

(def flatten-hsl
  "A transducer for unpacking a sequence of colors into a sequence of floating
  point numbers representing HSL values."
  (let [to-hsl (map tc/as-hsla)
        unpack (mapcat #(u/select-values % [:h :s :l]))]
    (comp to-hsl unpack)))

(def hues-only
  "A transducer for unpacking a sequence of colors into a sequence of their
  hues as floating-point numbers."
  (let [to-hsl (map tc/as-hsla)
        hues (map #(tc/hue %))]
    (comp to-hsl hues)))