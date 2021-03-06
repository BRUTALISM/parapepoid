(ns parapepoid.sketches.color-generation
  (:require [clojure.core.matrix :as matrix]
            [incanter.charts :as chart]
            [incanter.core :as inc]
            [parapepoid.color.core :as c]
            [parapepoid.color.learn :as l]
            [parapepoid.color.serialization :as cs]
            [parapepoid.nn.propagation :as p]
            [parapepoid.serialization :as s]
            [parapepoid.approach.core :as a]
            [parapepoid.approach.input-mappings :as ai]
            [parapepoid.approach.palette-generators :as ag]
            [quil.core :as q]
            [quil.middleware :as mid]
            [thi.ng.geom.vector :as v]
            [clojure.string :as str]))

(matrix/set-current-implementation :vectorz)

(def config {; Display params
             :number-of-samples 100
             :shape-radius 340
             :infinite-params {:hue 0.03
                               :saturation 1.0
                               :brightness 0.2}

             ; Approach params
             :number-of-colors 2
             :approach (a/approach (ag/hue-offset 0.1) ai/hues-only)
             ; TODO: Generate file name automatically based on approach params.
             :training-file "TR-I2-O1-hue-offset-0.1-hues-only.clj"

             ; NN params
             :network-params {:hidden-sizes [4]
                              :learning-rate 0.02
                              :batch-size 1
                              :epochs 100
                              :error-fn :cross-entropy}
             :approval-max-iterations 10
             :approval-threshold 0.7})

(defn color-to-shape [color]
  (let [y (/ (q/height) 2)
        xmax (q/width)
        radius (:shape-radius config)
        center (v/vec2 (int (rand xmax)) y)]
    {:color color
     :center center
     :radius (int (rand radius))}))

(defn display-errors [context errors]
  (let [xs (range (count errors))
        existing (:error-chart-window context)
        chart-window (inc/view (chart/xy-plot xs errors
                                              :x-label "Epoch"
                                              :y-label "Test error"))]
    (if existing (.dispose existing))
    (assoc context :error-chart-window chart-window)))

(defn generate-network [context]
  (let [file (:training-file config)
        approach (:approach config)
        [training test] (cs/read-data approach file 0.1)
        params (:network-params config)
        [network errors] (l/evaluate-hyper-params training test params)]
    (println "Done.")
    (-> context
        (display-errors errors)
        (assoc :network network))))

(defn palette-to-shapes [palette]
  (let [params (:infinite-params config)
        shapes (map color-to-shape (take (:number-of-samples config)
                                         (c/infinite-palette palette params)))]
    (reverse (sort-by :radius shapes))))

(defn palette-into-context [context palette]
  (assoc context
    :current-palette palette
    :shapes (palette-to-shapes palette)))

(defn approved-palette [context]
  (let [iterations (:approval-max-iterations config)
        num-colors (:number-of-colors config)
        approach (:approach config)
        network (:network context)]
    (loop [i 0
           palette (c/generate-base-palette approach num-colors)]
      (if (= i iterations)
        (do
          (println "NO COLORS APPROVED, random palette is shown.")
          (palette-into-context context palette))
        (let [result (->> (into [] (a/input-mapping (:approach config)) palette)
                          (p/propagate-forward network)
                          first)]
          (println "Propagation result: " result)
          (if (> result
                 (:approval-threshold config))
            (do
              (println "Accepting palette after " i " iterations.")
              (palette-into-context context palette))
            (recur (inc i) (c/generate-base-palette approach num-colors))))))))

(defn random-palette [context]
  (let [palette (c/generate-base-palette (:approach config)
                                         (:number-of-colors config))]
    (palette-into-context context palette)))

(defn- shape-to-rect [center radius]
  (let [x (- (:x center) (/ radius 2))
        y (- (:y center) (/ radius 2))]
    [x y radius radius]))

(defn save-training [context output]
  (let [palette (:current-palette context)
        training (conj (:training context) [palette output])]
    (s/write-training (:training-file config) training)
    (assoc context :training training)))

; Quil stuff

(defn setup []
  (q/frame-rate 10)
  (conj (-> {}
            random-palette
            generate-network)
        {:training (or (s/read-training (:training-file config)) [])}))

(defn update-context [context]
  context)

(defn draw [context]
  (q/background 240)
  (q/no-stroke)
  (doseq [shape (:shapes context)]
    (apply q/fill (c/as-rgb255-vec (:color shape)))
    (apply q/rect (shape-to-rect (:center shape) (:radius shape)))))

(defn key-pressed [context key-info]
  (case (:key key-info)
    :p (do
         (println "Saving palette into training data as NOT APPROVED.")
         (random-palette (save-training context 0)))
    :o (do
         (println "Palette is APPROVED, saving to training file.")
         (random-palette (save-training context 1)))
    :n (do
         (println "Generating new network.")
         (generate-network context))
    :a (do
         (println "Generating a new approved palette.")
         (approved-palette context))
    context))

(q/defsketch parapepoid
             :title "PARAPEPOID IS WATCHING YOU"
             :settings #(q/smooth 2)
             :setup setup
             :update update-context
             :draw draw
             :key-pressed key-pressed
             :size [800 400]
             :middleware [mid/fun-mode])
