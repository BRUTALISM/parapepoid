(ns parapepoid.approach.palette-generators
  (:require [thi.ng.color.core :as tc]
            [thi.ng.math.core :as m]))

; Various ways to generate color palettes.

(defn random-hsl
  ([] (tc/hsla (m/random) (m/random) (m/random)))
  ([_] (tc/hsla (m/random) (m/random) (m/random))))

(defn hue-offset [max-offset]
  (defn fun
    ([]
     (tc/random-rgb))
    ([c]
     (let [hue (tc/hue (tc/rotate-hue c (* max-offset (m/randnorm))))]
       (tc/hsla hue (tc/saturation c) (tc/luminance c)))))
  fun)