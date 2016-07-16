(ns parapepoid.nn.activation)

(defn sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn sigmoid-prime [x]
  (let [sigx (sigmoid x)]
    (* sigx (- 1.0 sigx))))
