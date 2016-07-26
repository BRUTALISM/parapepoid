(ns parapepoid.color
  (:require [thi.ng.color.core :as c]
            [thi.ng.geom.vector :as v]
            [thi.ng.math.core :as m]))

(defn- as-vec3 [c]
  (let [hsl (c/as-hsla c)]
    (v/vec3 (:h hsl) (:s hsl) (:l hsl))))

(defn diff
  "Calculates the difference between two colors, as a single number in the
  [0, 1] range."
  [c1 c2]
  (m/mag (m/- (as-vec3 c1)
              (as-vec3 c2))))