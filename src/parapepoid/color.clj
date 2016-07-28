(ns parapepoid.color
  (:require [thi.ng.color.core :as c]
            [thi.ng.geom.vector :as v]
            [thi.ng.math.core :as m]))

(defn- as-vec3 [c]
  (let [hsl (c/as-hsla c)]
    (v/vec3 (:h hsl) (:s hsl) (:l hsl))))

(defn as-rgb255-vec [color]
  (let [rgb (c/as-rgba color)]
    [(* (:r rgb) 255) (* (:g rgb) 255) (* (:b rgb) 255)]))

(defn diff
  "Calculates the difference between two colors, as a single number in the
  [0, 1] range."
  [c1 c2]
  (m/mag (m/- (as-vec3 c1)
              (as-vec3 c2))))

(defn random-hsl []
  (c/hsla (m/random) (m/random) (m/random)))

(defn random-pallete [count]
  (repeatedly count #(random-hsl)))

(defn next-color
  "Returns the next color in sequence for the given base colors and params."
  [colors params]
  (let [{:keys [hue saturation brightness]} params
        base (rand-nth colors)]
    (c/random-analog base hue saturation brightness)))

(defn infinite-palette
  "Creates a lazy infinite sequence of colors based off of the given base
  colors. The generation algorithm is configured using the params map."
  ([colors] (infinite-palette colors
                              {:hue 1.0 :saturation 1.0 :brightness 1.0}))
  ([colors params]
   (repeatedly #(next-color colors params))))
