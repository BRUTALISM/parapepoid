(ns parapepoid.approach.palette-generators
  (:require [thi.ng.color.core :as tc]
            [thi.ng.math.core :as m]))

; Various ways to generate color palettes.

(defn random-hsl
  ([] (tc/hsla (m/random) (m/random) (m/random)))
  ([_] (tc/hsla (m/random) (m/random) (m/random))))