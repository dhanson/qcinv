subroutine map2vlm( lmax, vmap, spin, vlm, npix )
  use healpix_types
  use alm_tools
  use pix_tools
  
  integer        :: lmax, spin, nside, npix, abspin
  double complex :: vmap(npix), vlm(0:(lmax+1)*(lmax+1)-1)
  double complex :: dalm(1:2, 0:lmax, 0:lmax)
  double precision :: dmap(npix, 1:2)
  
  abspin = abs(spin)
  
  if (spin .eq. abspin) then
     do i=1,npix
        dmap(i,1) =  real(vmap(i))
        dmap(i,2) =  imag(vmap(i))
     end do
  else
     do i=1,npix
        dmap(i,1) =  real(vmap(i))
        dmap(i,2) = -imag(vmap(i))
     end do
  end if

  nside = npix2nside(npix)
  call map2alm_spin(nside, lmax, lmax, abspin, dmap, dalm)
  
  if (spin .eq. abspin) then
     do l=0,lmax
        do m=0,l
           vlm(l*l+l-m) = -(-1)**m * conjg(dalm(1,l,m) - dalm(2,l,m)*(0,1))
           vlm(l*l+l+m) = -(dalm(1,l,m)+dalm(2,l,m)*(0,1))
        end do
     end do
  else
     do l=0,lmax
        do m=0,l
           vlm(l*l+l+m) = -(-1)**(spin) * (dalm(1,l,m)-dalm(2,l,m) *(0,1))
           vlm(l*l+l-m) = -conjg(dalm(1,l,m)+dalm(2,l,m)*(0,1))*(-1)**(m+spin)
        end do
     end do
  end if
end subroutine map2vlm

subroutine vlm2map( nside, vlm, spin, vmap, nvlm, lmax )
  use healpix_types
  use alm_tools
  use pix_tools
  
  integer        :: lmax, spin, nside, nvlm, abspin
  double complex :: vmap(12*nside*nside), vlm(0:((lmax+1)*(lmax+1)-1))
  double complex :: dalm(1:2, 0:lmax, 0:lmax)
  double precision :: dmap(12*nside*nside, 1:2)
  
  abspin = abs(spin)
  
  if (spin .eq. abspin) then
     do l=0,lmax
        do m=0,l
           dalm(1,l,m) = -0.5*(vlm(l*l+l+m) + (-1)**(m)*conjg(vlm(l*l+l-m)))
           dalm(2,l,m) = +0.5*(vlm(l*l+l+m) - (-1)**(m)*conjg(vlm(l*l+l-m)))*(0,1)
        end do
     end do
  else
     do l=0,lmax
        do m=0,l
           dalm(1,l,m) = -0.5*((-1)**(m) * conjg(vlm(l*l+l-m)) + vlm(l*l+l+m)) * (-1)**(spin)
           dalm(2,l,m) = +0.5*((-1)**(m) * conjg(vlm(l*l+l-m)) - vlm(l*l+l+m)) * (-1)**(spin) * (0,1)
        end do
     end do
  end if

  call alm2map_spin( nside, lmax, lmax, abspin, dalm, dmap)
  
  if (spin .eq. abspin) then
     do i=1,(12*nside*nside)
        vmap(i) =  dmap(i,1) + dmap(i,2)*(0,1)
     end do
  else
     do i=1,(12*nside*nside)
        vmap(i) =  dmap(i,1) - dmap(i,2)*(0,1)
     end do
  end if
end subroutine vlm2map
