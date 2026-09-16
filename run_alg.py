import numpy as np
import matplotlib.pyplot as plt
def parafac2_admm(X_list, rank, max_iter=500, tol=1e-5,prox_gA=lambda x, scale: x,prox_gB=lambda x, scale: x,prox_gD=lambda x, scale: x):
    K = len(X_list)
    I = X_list[0].shape[0]
    J_list = [X_k.shape[1] for X_k in X_list]
    R = rank

    A = np.abs(np.random.randn(I, R))
    B_list = [np.abs(np.random.randn(J_list[k], R)) for k in range(K)]
    D_vecs = [np.abs(np.random.randn(R)) for k in range(K)]



    # Biến auxiliary và dual cho A

    Z_A = A.copy()
    mu_A = np.zeros_like(A)
    ZB_list = [B.copy() for B in B_list]

    YB_list = [B.copy() for B in B_list]

    muZB_list = [np.zeros_like(B) for B in B_list]

    muDeltaB_list = [np.zeros_like(B) for B in B_list]

    P_list = [np.eye(J_list[k], R) for k in range(K)]

    Delta_B = np.eye(R)


    Z_D_list = [d.copy() for d in D_vecs]
    mu_D_list = [np.zeros_like(d) for d in D_vecs]

    norm_X_sq = sum(np.linalg.norm(X_k, 'fro')**2 for X_k in X_list)

    sse_list = []

    prev_sse = None
    for it in range(max_iter):

        # Update rho

        rho_A, rho_B, rho_D = update_rho(A, B_list, D_vecs)
        # Update B-mode

        B_list, ZB_list, YB_list, muZB_list, muDeltaB_list, P_list, Delta_B = update_B_mode(X_list, A, D_vecs, B_list, ZB_list, YB_list, muZB_list, muDeltaB_list, P_list, Delta_B, rho_B, prox_gB)



        # Update A-mode

        A, Z_A, mu_A = update_A_mode(X_list, B_list, D_vecs, Z_A, mu_A, rho_A, prox_gA)
        # Update D-mode
        D_vecs, Z_D_list, mu_D_list = update_D_mode(X_list, A, B_list, D_vecs, Z_D_list, mu_D_list, rho_D, prox_gD)
        sse = sum(np.linalg.norm(X_list[k] - A @ np.diag(D_vecs[k]) @ B_list[k].T, 'fro')**2 for k in range(K))
        rel_sse = sse / norm_X_sq
        sse_list.append(rel_sse)

        if prev_sse is not None:

            rel_change = abs(prev_sse - sse) / prev_sse
            if (it + 1) % 10 == 0 or it == 0:
                print(f"Iter {it+1:3d}/{max_iter} , SSE tương đối: {rel_sse:.6e}")

            if rel_change < tol:
                print(f"\nADMM hội tụ tại iter {it+1}")
                break  
        else:
            print(f"Iter {it+1:3d}/{max_iter} , SSE tương đối: {rel_sse:.6e}")

        prev_sse = sse

    return A, B_list, D_vecs, sse_list



if __name__ == "__main__":
    np.random.seed(42)

    # Toán tử Proximal không âm (Non-negativity constraint)

    def prox_nonnegative(x, scale=1.0):
        return np.maximum(0, x)

    # Chuẩn hóa để tránh scale ambiguity
    def prox_normalize(x, scale=1.0):
        norm = np.linalg.norm(x, axis=0, keepdims=True)
        norm[norm < 1.0] = 1.0
        return x / norm



    # Chuẩn L1 đếm số phần tử khác 0, giúp data thưa
    def prox_l1(x, scale=1.0, l1_reg =0.01):
        threshold = l1_reg * scale
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    # Chuẩn L2 giảm trọng số, ép trọng số giảm về 0 (nhưng không bằng 0) để tìm đường cong phù hợp phân bố data

    def prox_l2(x, scale=1.0, l2_reg=0.01):
        return x / (1.0 + l2_reg * scale)



    def prox_identity(x, scale=1.0):
        return x

    def prox_nonnegative_l2(x, scale=1.0, l2_reg=0.01):
        scaled_x = x / (1.0 + l2_reg * scale)
        return np.maximum(0, scaled_x)
    

    
    K, I, R = 5, 20, 3

    J_list = [30, 35, 25, 40, 28]  

   

    A_true = np.maximum(0, np.random.randn(I, R))
    B_true = [np.maximum(0, np.random.randn(J_list[k], R)) for k in range(K)]
    D_true = [np.random.uniform(0.5, 1.5, R) for k in range(K)]



    print("\nRiel A:\n",A_true)
    for k in range(K):
            print(f"\nRiel B[{k}]:\n", B_true[k])
            print(f"Riel D[{k}]:\n", D_true[k])



    # X_k = A * diag(d_k) * B_k^T + noise
    sigmas = [0.1, 0.01, 0.001, 0.0001, 0.00000001, 0]
    res = {}
    fms_res = {}
    for sigma in sigmas:
      X_list = []
      for k in range(K):
        X_cl = A_true @ np.diag(D_true[k]) @ B_true[k].T
        nhieu_gauss = np.random.normal(loc=0.0, scale=sigma, size=(I, J_list[k]))
        X_list.append(X_cl + nhieu_gauss)

      
      A_est, B_est, D_est, sse_list = parafac2_admm(
        X_list,
        rank=R,
        max_iter=300,
        prox_gA=prox_normalize, 
        prox_gB=prox_identity,
        prox_gD=prox_identity
      )
      res[sigma] = sse_list  

      fms_overall, fms_A, fms_B, fms_D = parafac2_fms(A_true, B_true, D_true, A_est, B_est, D_est)

      fms_res[sigma] = {
          "Overall" : fms_overall, 
          "A": fms_A, 
          "B": fms_B, 
          "D": fms_D,
      }
                        
    print("\nEstimated A:\n", A_est)

    for k in range(K):
        print(f"\nEstimated B[{k}]:\n", B_est[k])
        print(f"Estimated D[{k}]:\n", D_est[k])


    print("\n")
    for sigma, scores in fms_res.items():
        print(f"Sigma = {sigma:<7}, FMS tb: {scores['Overall']:.4f}, FMS_A: {scores['A']:.4f}, FMS_B: {scores['B']:.4f}, FMS_D: {scores['D']:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for sigma, sse_list in res.items():
        plt.semilogy(sse_list, label=f"sigma={sigma}")
        
    plt.xlabel("Iteration")
    plt.ylabel("Error (Log scale)")
    plt.title("Convergence (Relative SSE)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)  

    plt.subplot(1, 2, 2)
    sigmas_str = [str(s) for s in sigmas]
    fms_overall_vals = [fms_res[s]["Overall"] for s in sigmas]
    fms_A_vals = [fms_res[s]["A"] for s in sigmas]
    fms_B_vals = [fms_res[s]["B"] for s in sigmas]
    fms_D_vals = [fms_res[s]["D"] for s in sigmas]

    plt.plot(sigmas_str, fms_overall_vals, "o-", label="FMS TB", linewidth=2)
    plt.plot(sigmas_str, fms_A_vals, "s--", label="FMS A")
    plt.plot(sigmas_str, fms_B_vals, "^--", label="FMS B")
    plt.plot(sigmas_str, fms_D_vals, "m^--", label="FMS D")

    plt.xlabel("Noise (Sigma)")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()